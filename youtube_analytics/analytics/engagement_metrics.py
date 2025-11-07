import os
import json
import argparse
from tqdm import tqdm
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta


def _to_int(val: Any) -> int:
    if val is None:
        return 0
    try:
        return int(val)
    except Exception:
        try:
            return int(str(val).replace(",", ""))
        except Exception:
            return 0


def parse_iso_date(d: Optional[str]) -> Optional[datetime]:
    if not d:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(d, fmt)
        except Exception:
            continue
    # try flexible parse fallback (very permissive)
    try:
        return datetime.fromisoformat(d)
    except Exception:
        return None


def compute_video_metrics(video_data: Dict) -> Dict:
    view_count = _to_int(video_data.get("view_count") or 0)
    # prefer explicit comment_count field; else fall back to comments list length
    comment_count = _to_int(
        video_data.get("comment_count")
        or (
            len(video_data.get("comments", []))
            if video_data.get("comments") is not None
            else 0
        )
    )
    like_count = _to_int(video_data.get("like_count") or 0)

    metrics = {
        "view_count": view_count,
        "comment_count": comment_count,
        "like_count": like_count,
        "comment_rate": None,
        "like_rate": None,
        "engagement_rate": None,
        # engaged_like_ratio: how many comments per like (None when like_count == 0)
        "engaged_like_ratio": None,
    }

    if view_count > 0:
        metrics["comment_rate"] = comment_count / view_count
        metrics["like_rate"] = like_count / view_count
        metrics["engagement_rate"] = (comment_count + like_count) / view_count

    if like_count > 0:
        metrics["engaged_like_ratio"] = comment_count / like_count

    return metrics


def analyze_channel_engagement(
    channel_id: str,
    data_root: str = "data",
    days: Optional[int] = None,
    since: Optional[str] = None,
) -> None:
    channel_dir = os.path.join(data_root, channel_id)
    if not os.path.exists(channel_dir):
        print(f"Channel directory not found: {channel_dir}")
        return

    # compute date cutoff
    cutoff: Optional[datetime] = None
    if since:
        cutoff = parse_iso_date(since)
        if cutoff is None:
            print(f"Could not parse --since date: {since}")
            return
    elif days is not None:
        cutoff = datetime.utcnow() - timedelta(days=days)

    channel_metadata_path = os.path.join(channel_dir, "channel_metadata.json")
    channel_metadata = {}
    if os.path.exists(channel_metadata_path):
        try:
            with open(channel_metadata_path, "r", encoding="utf-8") as f:
                channel_metadata = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load channel metadata: {e}")

    per_video_summaries: List[Dict] = []

    # Aggregation accumulators **from files**
    total_views = 0
    total_comments = 0
    total_likes = 0
    videos_processed = 0

    for filename in tqdm(
        sorted(os.listdir(channel_dir)), desc=f"Processing channel {channel_id}"
    ):
        if filename == "channel_metadata.json" or not filename.endswith(".json"):
            continue

        filepath = os.path.join(channel_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                video_data = json.load(f)

            # decide whether to include this video based on published_at and cutoff
            pub = parse_iso_date(video_data.get("published_at"))
            if cutoff is not None and pub is not None:
                # include only if published_at >= cutoff
                if pub < cutoff:
                    # skip older video
                    continue
            elif cutoff is not None and pub is None:
                # if we can't parse the date, skip with a warning
                print(f"Skipping {filename}: missing or unparsable published_at")
                continue

            metrics = compute_video_metrics(video_data)

            # attach engagement_metrics into video_data and persist
            video_data["engagement_metrics"] = {
                "view_count": metrics["view_count"],
                "comment_count": metrics["comment_count"],
                "like_count": metrics["like_count"],
                "comment_rate": (
                    round(metrics["comment_rate"], 6)
                    if metrics["comment_rate"] is not None
                    else None
                ),
                "like_rate": (
                    round(metrics["like_rate"], 6)
                    if metrics["like_rate"] is not None
                    else None
                ),
                "engagement_rate": (
                    round(metrics["engagement_rate"], 6)
                    if metrics["engagement_rate"] is not None
                    else None
                ),
                "engaged_like_ratio": (
                    round(metrics["engaged_like_ratio"], 6)
                    if metrics["engaged_like_ratio"] is not None
                    else None
                ),
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(video_data, f, indent=2, ensure_ascii=False)

            # update aggregates
            total_views += metrics["view_count"]
            total_comments += metrics["comment_count"]
            total_likes += metrics["like_count"]
            videos_processed += 1

            per_video_summaries.append(
                {
                    "video_id": video_data.get("video_id")
                    or filename.replace(".json", ""),
                    "title": video_data.get("title", ""),
                    "published_at": video_data.get("published_at"),
                    "metrics": video_data["engagement_metrics"],
                }
            )

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # channel-level metrics computed from the included videos only
    channel_engagement: Dict = {}
    channel_engagement["videos_included"] = videos_processed
    channel_engagement["total_view_count"] = total_views
    channel_engagement["total_comment_count"] = total_comments
    channel_engagement["total_like_count"] = total_likes

    if total_views > 0:
        channel_engagement["overall_comment_rate"] = round(
            total_comments / total_views, 6
        )
        channel_engagement["overall_like_rate"] = round(total_likes / total_views, 6)
        channel_engagement["overall_engagement_rate"] = round(
            (total_comments + total_likes) / total_views, 6
        )
    else:
        channel_engagement["overall_comment_rate"] = None
        channel_engagement["overall_like_rate"] = None
        channel_engagement["overall_engagement_rate"] = None

    # engaged-like ratio at channel level: aggregate comment/like where meaningful
    if total_likes > 0:
        channel_engagement["overall_engaged_like_ratio"] = round(
            total_comments / total_likes, 6
        )
    else:
        channel_engagement["overall_engaged_like_ratio"] = None

    # also compute simple means across videos (excluding None values)
    comment_rates = [
        v["metrics"]["comment_rate"]
        for v in per_video_summaries
        if v["metrics"]["comment_rate"] is not None
    ]
    like_rates = [
        v["metrics"]["like_rate"]
        for v in per_video_summaries
        if v["metrics"]["like_rate"] is not None
    ]
    engagement_rates = [
        v["metrics"]["engagement_rate"]
        for v in per_video_summaries
        if v["metrics"]["engagement_rate"] is not None
    ]
    engaged_like_ratios = [
        v["metrics"]["engaged_like_ratio"]
        for v in per_video_summaries
        if v["metrics"]["engaged_like_ratio"] is not None
    ]

    def _mean(lst: List[float]) -> Optional[float]:
        return round(sum(lst) / len(lst), 6) if lst else None

    channel_engagement["mean_comment_rate_across_videos"] = _mean(comment_rates)
    channel_engagement["mean_like_rate_across_videos"] = _mean(like_rates)
    channel_engagement["mean_engagement_rate_across_videos"] = _mean(engagement_rates)
    channel_engagement["mean_engaged_like_ratio_across_videos"] = _mean(
        engaged_like_ratios
    )

    # attach to channel_metadata and save
    channel_metadata["engagement_metrics"] = channel_engagement
    channel_metadata["per_video_engagement_summary"] = per_video_summaries

    try:
        with open(channel_metadata_path, "w", encoding="utf-8") as f:
            json.dump(channel_metadata, f, indent=2, ensure_ascii=False)
        print(
            f"Wrote channel metadata with engagement metrics to: {channel_metadata_path}"
        )
    except Exception as e:
        print(f"Failed to write channel metadata file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute engagement metrics for a channel's video JSONs (aggregated from files)"
    )
    parser.add_argument(
        "channel_id", help="The YouTube channel ID (folder name) to analyze"
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory for data files (default: data)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Analyze only videos published within the last N days",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Analyze only videos published on or after this date (YYYY-MM-DD)",
    )

    args = parser.parse_args()

    analyze_channel_engagement(
        channel_id=args.channel_id,
        data_root=args.data_root,
        days=args.days,
        since=args.since,
    )
