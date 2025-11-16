# Project overview

This is a small, self-contained YouTube analytics prototype intended to demonstrate data-science + UX-research skills.

It:

* fetches channel metadata, recent videos and top-level comments (`youtube_analytics/data/channel_data.py`),
* computes per-video and per-channel engagement metrics and persists them into the saved JSONs (`youtube_analytics/analytics/engagement_metrics.py`), and
* runs comment-level NLP: sentiment tagging (`youtube_analytics/nlp/sentiment.py`) and zero-shot topic classification (`youtube_analytics/nlp/topic_classification.py`).

Data is saved under `data/<channel_id>/` as JSON files. `channel_metadata.json` sits alongside per-video `<video_id>.json` files.

---

# What already works (implemented features)

## Data collection

* `fetch_channel_data(channel_url, num_videos=10, num_comments=50, data_dir='./data/', ollama_model=None)`

  * Resolves a channel from username / channel URL / handle.
  * Fetches channel metadata and writes `channel_metadata.json`.
  * Retrieves the last `N` videos from the channel uploads playlist (titles, published dates).
  * Fetches video metadata (views / likes / comments counts).
  * Attempts to fetch transcripts via `youtube_transcript_api` (multiple language list). If `ollama_model` is provided and Ollama is available, summarizes transcripts via Ollama.
  * Fetches top-level comments (paginated up to the requested number).
  * Writes one JSON file per video: `{video_metadata, transcript, summary, comments}`.

Files: `youtube_analytics/data/channel_data.py`

---

## Engagement metrics

* `analyze_channel_engagement(channel_id, data_root='data', days=None, since=None)`

  * Loads video JSONs from `data/<channel_id>/`.
  * Computes per-video metrics: `view_count`, `comment_count`, `like_count`, `comment_rate`, `like_rate`, `engagement_rate`, `engaged_like_ratio`.
  * Writes those metrics back into each `<video_id>.json` under `engagement_metrics`.
  * Aggregates channel-level metrics and writes them into `channel_metadata.json` as `engagement_metrics` & `per_video_engagement_summary`.

Files: `youtube_analytics/analytics/engagement_metrics.py`

---

## Sentiment analysis (comment-level)

* `analyze_channel_sentiment(channel_id, data_root='data', batch_size=64)`

  * Loads a multilingual HuggingFace classification model (`AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual`) and tokenizes comments in batches.
  * Adds a `sentiment` field to each comment with probability scores for Negative / Neutral / Positive.
  * Rewrites video JSONs with enriched comments.

Files: `youtube_analytics/nlp/sentiment.py`

Notes: runs on CPU or CUDA if available. Large channels/comments will be slow on CPU.

---

## Topic classification (zero-shot)

* `zero_shot_classify_channel(channel_id, data_root='data', model_name='joeddav/xlm-roberta-large-xnli', ...)`

  * Uses Hugging Face zero-shot pipeline to assign one or multiple topic labels (categories defined in `CATEGORY_DEFINITIONS`) to each comment.
  * Produces a `results` structure (video-level topic distributions, per-comment `assigned_topics`) which can be persisted using `save_results_to_files(...)`.
  * `save_results_to_files` merges classification back into original video JSONs (optionally overwriting or writing new files with a suffix) and updates `channel_metadata.json` with `global_topic_distribution`.

Files: `youtube_analytics/nlp/topic_classification.py`

---

# Data layout (what the `data/` folder looks like)

```
data/
  <channel_id>/
    channel_metadata.json    # channel summary + (after analysis) engagement_metrics, global_topic_distribution
    <video_id>.json          # per-video metadata: title, description, published_at, view_count, like_count, comment_count,
                             # plus transcript, summary, comments (each comment is a dict), and after runs:
                             # engagement_metrics, per-comment sentiment, per-comment assigned_topics (if applied)
```

---

# How to run (quick start)

> **Precondition**: Create `youtube_analytics/config.py` containing `API_KEY = "<YOUR_YOUTUBE_API_KEY>"` (do not check credentials into git).

1. create and activate a virtual environment. TODO: Share the conda enviroment for installation.

2. fetch channel data (example usage inside a Python REPL or script):

```python
from youtube_analytics.data.channel_data import fetch_channel_data

# fetch 10 latest videos, 50 comments each, save under ./data/
fetch_channel_data("https://www.youtube.com/@somechannel", num_videos=10, num_comments=50, data_dir="./data/", ollama_model=None)
```

3. compute engagement metrics:

```python
from youtube_analytics.analytics.engagement_metrics import analyze_channel_engagement

# compute metrics for the channel and update JSONs
analyze_channel_engagement(channel_id="UCXXXX...", data_root="data")
```

4. run sentiment tagging (requires torch + huggingface model download):

```python
from youtube_analytics.nlp.sentiment import analyze_channel_sentiment

analyze_channel_sentiment(channel_id="UCXXXX...", data_root="data", batch_size=64)
```

5. run zero-shot topic classification:

```python
from youtube_analytics.nlp.topic_classification import zero_shot_classify_channel, save_results_to_files

results = zero_shot_classify_channel(channel_id="UCXXXX...", data_root="data")
save_results_to_files(results, data_root="data", overwrite=False, output_suffix="_classified")
```

If you prefer command-line wrappers you can add small driver scripts that call these functions.

---

Notes:

* Hugging Face models will be downloaded the first time they are used; ensure your environment allows this. Models can be large (~several hundred MBs to multiple GBs).
* For large batches, running on GPU is recommended. The sentiment and topic pipelines can be slow on CPU.
* `ollama` usage is optional — the code checks if it's available.
* 
---

# Configuration & secrets

* `youtube_analytics/config.py` (not included): must define `API_KEY = "<YOUR_YOUTUBE_API_KEY>"`.
* Do **not** commit your API key. Add `config.py` to `.gitignore` (or set the key via environment management in production scripts).

Optional:

* Ollama: if you want transcript summarization via Ollama, provide the model name to `fetch_channel_data(..., ollama_model="modelname")` and ensure local Ollama is installed and accessible.

---

# Output examples (what the JSONs will contain)

Each `<video_id>.json` will contain keys like:

```json
{
  "video_id": "abc123",
  "title": "...",
  "description": "...",
  "published_at": "2024-05-01",
  "view_count": "12345",
  "like_count": "678",
  "comment_count": "90",
  "transcript": "....",
  "summary": "<summary>...</summary>",
  "comments": [
    {
      "comment_id": "abcd",
      "video_id": "abc123",
      "author": "User",
      "date": "2024-05-02",
      "likes": 5,
      "comment": "Great video!",
      "num_replies": 1,
      "sentiment": { "Negative": 0.00, "Neutral": 0.10, "Positive": 0.90 },
      "assigned_topics": [{ "label": "Video Content", "score": 0.87 }]
    }
  ],
  "engagement_metrics": {
    "view_count": 12345,
    "comment_count": 90,
    "like_count": 678,
    "comment_rate": 0.00729,
    "like_rate": 0.0549,
    "engagement_rate": 0.0622,
    "engaged_like_ratio": 0.1328
  }
}
```

`channel_metadata.json` will be augmented with `engagement_metrics` and `global_topic_distribution`.

---

# Known limitations & edge cases

* **Shorts vs full videos**: the code currently does not explicitly distinguish YouTube Shorts from regular uploads; this is on your TODO list. Shorts may have different engagement dynamics.
* **Comment pages**: `get_comments` requests up to 100 per request; code attempts pagination but is conservative—comments beyond the requested number may need extra handling.
* **Rate limits & quotas**: Google API quotas apply. For large-scale collection you must handle quota-exhaustion & exponential backoff. The code prints and continues on many exceptions but does not implement a robust retry/backoff strategy yet.
* **Multilingual comments**: zero-shot classification and sentiment are multilingual-friendly, but performance varies by language and domain.
* **Ollama**: optional and only used if provided; the code attempts to pull and test the model.

---

# Planned features / roadmap (from `todo.txt`)

1. **Differentiate Shorts vs full videos**

   * Use video duration (from `contentDetails`) or `shorts` metadata to tag files and analyze separately.

2. **Topic modeling segmented by sentiment**

   * For positive/negative comment subsets: analyze topic modeling results to find what people praise vs criticise.

3. **Weighted topic contribution**

   * Weight each comment's influence on topic statistics by `likes` and (possibly) `num_replies` (proposed weight formula in TODO).

4. **Sentiment/topic weighted metrics**

   * Build composite metrics per-video: e.g., weighted sentiment score, topic-weighted engagement.

5. **Correlation analysis**

   * Explore correlations between sentiment/topics and engagement metrics (views/likes/comments/retention if/when available).

6. **Toxicity & humour detection**

   * Evaluate classifiers for toxicity and humour detection (off-the-shelf models or custom fine-tuning).

7. **UX / reporting**

   * Add a small Dash/Streamlit frontend with interactive visualizations summarizing: top topics, sentiment trends, engagement per video, and per-topic engagement.

8. **Robust productionization**

   * Add logging, retries/backoff, rate-limit handling, small CLI wrappers, and unit tests for parsing/aggregation.
