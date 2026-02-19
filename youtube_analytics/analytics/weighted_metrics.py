import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm


def compute_comment_weight(
    likes: int, replies: int, like_weight: float, reply_weight: float
) -> float:
    """
    Computes the weight of a comment using a logarithmic scale to prevent
    viral outliers from entirely dominating the metrics.
    Formula: Weight = 1 + like_weight * log(1 + num_likes) + reply_weight * log(1 + num_replies)
    """
    # math.log1p(x) is equivalent to math.log(1 + x)
    return (
        1.0 + (like_weight * math.log1p(likes)) + (reply_weight * math.log1p(replies))
    )


def calculate_weighted_metrics(
    channel_id: str,
    data_root: str = "data",
    like_weight: float = 1.0,
    reply_weight: float = 1.5,  # Slightly higher default for replies as they show active engagement
):
    channel_dir = Path(data_root) / channel_id
    if not channel_dir.exists():
        print(f"Channel directory not found: {channel_dir}")
        return

    # Get list of video files (excluding metadata)
    video_files = [
        p for p in channel_dir.glob("*.json") if p.name != "channel_metadata.json"
    ]

    if not video_files:
        print("No video files found.")
        return

    print(
        f"Computing weighted metrics for {len(video_files)} videos in channel {channel_id}..."
    )

    for video_file in tqdm(video_files, desc="Processing videos"):
        try:
            with open(video_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            comments = data.get("comments", [])
            if not comments:
                continue

            total_video_weight = 0.0

            # Accumulators for weighted sums
            sentiment_weighted_sum = {}
            topic_weighted_sum = {}
            topic_sentiment_weighted_sum = (
                {}
            )  # To hold sentiment specifically for each topic

            # 1. Compute weights per comment and accumulate sums
            for comment in comments:
                likes = int(comment.get("likes", 0))
                num_replies = int(comment.get("num_replies", 0))

                w_i = compute_comment_weight(
                    likes, num_replies, like_weight, reply_weight
                )
                comment["weight"] = round(w_i, 3)
                total_video_weight += w_i

                # Accumulate Global Video Sentiment
                if "sentiment" in comment and isinstance(comment["sentiment"], dict):
                    for label, score in comment["sentiment"].items():
                        sentiment_weighted_sum[label] = sentiment_weighted_sum.get(
                            label, 0.0
                        ) + (score * w_i)

                # Accumulate Topics and Topic-Specific Sentiment
                if "assigned_topics" in comment and isinstance(
                    comment["assigned_topics"], list
                ):
                    for topic_entry in comment["assigned_topics"]:
                        label = topic_entry.get("label")
                        topic_score = topic_entry.get("score", 0.0)

                        if label:
                            # Topic Impact = Engagement Weight * Model Confidence
                            impact = topic_score * w_i
                            topic_weighted_sum[label] = (
                                topic_weighted_sum.get(label, 0.0) + impact
                            )

                            # Initialize topic-specific sentiment dictionary if not exists
                            if label not in topic_sentiment_weighted_sum:
                                topic_sentiment_weighted_sum[label] = {
                                    "Negative": 0.0,
                                    "Neutral": 0.0,
                                    "Positive": 0.0,
                                    "_total_impact": 0.0,
                                }

                            topic_sentiment_weighted_sum[label][
                                "_total_impact"
                            ] += impact

                            # Distribute the impact across the comment's sentiment
                            if "sentiment" in comment and isinstance(
                                comment["sentiment"], dict
                            ):
                                for sent_label, sent_score in comment[
                                    "sentiment"
                                ].items():
                                    topic_sentiment_weighted_sum[label][sent_label] += (
                                        sent_score * impact
                                    )

            # 2. Calculate final weighted scores (Normalized)
            weighted_sentiment = {}
            weighted_topics = {}
            topic_specific_sentiment = {}

            # Normalize Video Sentiment
            if total_video_weight > 0:
                for label, total_score in sentiment_weighted_sum.items():
                    weighted_sentiment[label] = round(
                        total_score / total_video_weight, 4
                    )

            # Normalize Topic Share (Relative Dominance)
            total_topic_impact = sum(topic_weighted_sum.values())
            if total_topic_impact > 0:
                for label, total_score in topic_weighted_sum.items():
                    weighted_topics[label] = round(total_score / total_topic_impact, 4)

            # Normalize Topic-Specific Sentiment
            for label, sent_data in topic_sentiment_weighted_sum.items():
                t_impact = sent_data.pop("_total_impact")  # Remove the meta-key
                if t_impact > 0:
                    topic_specific_sentiment[label] = {
                        s_label: round(s_score / t_impact, 4)
                        for s_label, s_score in sent_data.items()
                    }

            # 3. Write back to video data
            if "weighted_metrics" not in data:
                data["weighted_metrics"] = {}

            data["weighted_metrics"]["sentiment"] = weighted_sentiment
            data["weighted_metrics"]["topic_dominance"] = weighted_topics
            data["weighted_metrics"][
                "topic_specific_sentiment"
            ] = topic_specific_sentiment

            # Save the file
            with open(video_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute weighted metrics (sentiment, topics) for YouTube comments based on likes and replies."
    )
    parser.add_argument(
        "channel_id", help="The YouTube channel ID (folder name) to analyze"
    )
    parser.add_argument(
        "--data-root", default="data", help="Root directory for data files"
    )
    parser.add_argument(
        "--like-weight",
        type=float,
        default=1.0,
        help="Scaling coefficient for like count",
    )
    parser.add_argument(
        "--reply-weight",
        type=float,
        default=1.5,
        help="Scaling coefficient for reply count",
    )

    args = parser.parse_args()

    calculate_weighted_metrics(
        channel_id=args.channel_id,
        data_root=args.data_root,
        like_weight=args.like_weight,
        reply_weight=args.reply_weight,
    )
