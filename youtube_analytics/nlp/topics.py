import os
import json
import time
import requests
from typing import List, Optional
from tqdm import tqdm

CATEGORY_DEFINITIONS = {
    "Video Content": "Comments about the topic or subject matter of the video itself.",
    "Production Quality": "Feedback on audio, video quality, editing, or overall presentation.",
    "Information Value": "Remarks on the educational content, usefulness of the information, or what was learned.",
    "Creator Personality": "Comments about the host's style, humor, or personal attributes.",
    "Expertise & Credibility": "Discussion of the creator's knowledge, authority, or trustworthiness on the topic.",
    "Community & Discussion": "Interaction between commenters or discussion about the channel's community.",
    "Requests & Suggestions": "Ideas for future video topics or suggestions for improving the channel.",
    "Personal Stories & Experiences": "Viewers sharing their own related anecdotes or life experiences.",
    "Technical Issues": "Problems with video playback, audio sync, or YouTube platform bugs.",
    "Channel Management": "Comments about upload frequency, video length, titles, or thumbnails.",
    "Comparisons": "Comparing the video's topic or the creator to something or someone else.",
    "Off-Topic & Unrelated": "Spam, irrelevant, or nonsensical comments.",
}

CATEGORY_LABELS = list(CATEGORY_DEFINITIONS.keys())

# LM Studio config via environment variables (customize as needed)
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1/completions")
LM_DEFAULT_MAX_TOKENS = 64


def build_prompt(comment: str, video_title: str) -> str:
    """
    Builds a clear instruction prompt for the language model.
    Includes video title for context and instructs the model to reply
    with exactly one category name from the taxonomy.
    """
    taxonomy = "\n".join(
        f"- **{label}**: {desc}" for label, desc in CATEGORY_DEFINITIONS.items()
    )
    prompt = (
        "You are an expert YouTube comment analyst. Your task is to classify a comment into a single category based on the provided video title and a fixed taxonomy.\n\n"
        f"**Video Title:**\n{video_title}\n\n"
        f"**Comment to Classify:**\n{comment}\n\n"
        "**Taxonomy and Definitions:**\n"
        f"{taxonomy}\n\n"
        "**Instructions:**\n"
        "1. Read the video title and the comment carefully to understand the context.\n"
        "2. Review the taxonomy and choose the SINGLE category that best describes the main point of the comment.\n"
        "3. Respond with ONLY the category name from the list. Do not add explanations or any other text."
    )
    return prompt


def query_lm_studio(
    prompt: str,
    max_tokens: int = LM_DEFAULT_MAX_TOKENS,
) -> Optional[str]:
    """
    Send a prompt to LM Studio and return the raw text output from the model.
    Uses simple retry/backoff on failures. Returns None on permanent failure.
    The payload fields are intentionally generic; adapt them if your LM Studio endpoint expects different keys.
    """
    headers = {"Content-Type": "application/json"}

    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
    }

    try:
        r = requests.post(LM_STUDIO_URL, json=payload, headers=headers)
        r.raise_for_status()
        # Expecting JSON like: {"choices":[{"text":"..."}], ...}
        data = r.json()
        # Flexible parsing: common keys are choices[0].text or output_text or result
        text = None
        if isinstance(data, dict):
            if (
                "choices" in data
                and isinstance(data["choices"], list)
                and data["choices"]
            ):
                text = (
                    data["choices"][0].get("text")
                    or data["choices"][0].get("message")
                    or None
                )
            elif "text" in data:
                text = data.get("text")
            elif "output" in data:
                text = data.get("output")
            elif "result" in data:
                text = data.get("result")
        # As a fallback, try to interpret the raw response body as text
        if text is None:
            try:
                text = r.text
            except Exception:
                text = None

        if text is not None:
            return str(text).strip()
        else:
            # Null-ish response; treat as transient error
            raise ValueError("LM Studio returned no textual output.")
    except Exception as e:
        print(f"Error querying LM Studio: {str(e)}")
    return None


def parse_category_from_model_output(output: Optional[str]) -> str:
    """
    Given raw model output, pick the best matching category label.
    Attempts:
    1) Exact case-insensitive equality with a label.
    2) Label is a substring of output (case-insensitive).
    3) Fuzzy fallback: check label words individually (rare).
    4) Final fallback -> "Off-Topic & Unrelated"
    """
    if not output:
        return "Off-Topic & Unrelated"

    out = output.strip().strip('"').strip("'").strip()
    # 1) exact match (case-insensitive)
    for label in CATEGORY_LABELS:
        if out.lower() == label.lower():
            return label

    # 2) substring match
    for label in CATEGORY_LABELS:
        if label.lower() in out.lower():
            return label

    # 3) try the reverse: if model returned something like "The category is: Video Content"
    for label in CATEGORY_LABELS:
        if label.lower().split()[0] in out.lower():
            return label

    # 4) fallback
    return "Off-Topic & Unrelated"


def process_channel_directory(channel_dir: str, data_root="data"):
    """
    Processes all video JSON files in a channel directory, classifying each comment
    using an LM Studio-based model.
    """
    # Normalize and resolve the channel directory path
    if not os.path.isabs(channel_dir):
        channel_dir = os.path.join(data_root, channel_dir)
    channel_dir = os.path.abspath(channel_dir)

    if not os.path.exists(channel_dir):
        print(f"Channel directory not found: {channel_dir}")
        return

    video_files = [
        f
        for f in os.listdir(channel_dir)
        if f.endswith(".json") and f != "channel_metadata.json"
    ]

    for filename in tqdm(video_files, desc="Processing videos"):
        filepath = os.path.join(channel_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_title = data.get("title", "Untitled Video")
        comments = data.get("comments", [])

        if not comments:
            continue

        # Classify each comment individually, showing progress
        for comment_obj in tqdm(
            comments,
            desc=f"Classifying comments for '{video_title[:30]}...'",
            leave=False,
        ):
            comment_text = comment_obj.get("comment", "").strip()
            if not comment_text:
                comment_obj["topic"] = "Off-Topic & Unrelated"
                continue

            prompt = build_prompt(comment_text, video_title)
            raw_output = query_lm_studio(prompt)
            topic = parse_category_from_model_output(raw_output)
            comment_obj["topic"] = topic

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
