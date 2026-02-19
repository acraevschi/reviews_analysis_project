from pathlib import Path
import json
from typing import List, Dict, Any, Optional, Tuple
from transformers import pipeline
import torch
import argparse
import sys


CATEGORY_DEFINITIONS = {
    "Video Content": "Comments about the topic or subject matter of the video itself.",
    "Production Quality": "Feedback on audio, video quality, editing, or overall presentation.",
    "Information Value": "Remarks on the educational content, usefulness of the information, or what was learned.",
    "Personality": "Comments about the host's or guest's style, humor, or personal attributes.",
    "Requests & Suggestions": "Ideas for future video topics or suggestions for improving the channel.",
    "Personal Stories": "Viewers sharing their own related anecdotes or life experiences.",
    "Technical Problems": "Problems with video playback, audio sync, or YouTube platform bugs.",
    "Off-Topic & Unrelated": "Spam, irrelevant, or nonsensical comments.",
}


def _get_device() -> int:
    """Return device index: 0..n for CUDA, -1 for CPU."""
    return 0 if torch.cuda.is_available() else -1


def _load_zero_shot_pipeline(model_name: str = "joeddav/xlm-roberta-large-xnli"):
    """
    Load the HuggingFace zero-shot pipeline with a multilingual NLI model.
    Using XLM-R large XNLI model is a good multilingual choice.
    """
    device = _get_device()
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )
    return classifier


def _iter_channel_comment_files(channel_dir: Path):
    """
    Yield JSON file paths in the channel directory, skipping channel_metadata.json.
    Only load files that look like video JSONs (we assume other files are ignored).
    """
    for p in sorted(channel_dir.glob("*.json")):
        if p.name == "channel_metadata.json":
            continue
        yield p


def _load_comments_from_video_json(
    video_json_path: Path,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Load comments list from a video JSON. Returns (video_id / filename, comments_list).
    Expects the file structure you specified: top-level 'comments' array.
    """
    data = json.loads(video_json_path.read_text(encoding="utf-8"))
    comments = data.get("comments", [])
    # preserve video id if present, else derive from filename
    video_id = data.get("video_id") or video_json_path.stem
    return video_id, comments


def _clean_comment_text(text: str) -> str:
    """
    Minimal cleaning: strip, remove excessive whitespace. Keep language-sensitive content intact.
    Avoid aggressive tokenization/stopword removal because multilingual.
    """
    if text is None:
        return ""
    return " ".join(text.strip().split())


def zero_shot_classify_channel(
    channel_id: str,
    data_root: str = "data",
    model_name: str = "joeddav/xlm-roberta-large-xnli",
    hypothesis_template: str = "This YouTube comment discusses {}.",
    multi_label: bool = True,
    score_threshold: float = 0.35,
    top_k: Optional[int] = 3,
    augment_label_with_definition: bool = True,
    batch_size: int = 8,
) -> Dict[str, Any]:
    classifier = _load_zero_shot_pipeline(model_name=model_name)

    channel_dir = Path(data_root) / channel_id
    if not channel_dir.exists() or not channel_dir.is_dir():
        raise FileNotFoundError(f"Channel directory not found: {channel_dir}")

    candidate_labels = list(CATEGORY_DEFINITIONS.keys())
    if augment_label_with_definition:
        candidate_labels_aug = [
            f"{label} — {CATEGORY_DEFINITIONS[label]}" for label in candidate_labels
        ]
    else:
        candidate_labels_aug = candidate_labels

    results = {
        "channel_id": channel_id,
        "videos": {},
        "global_topic_distribution": {k: 0.0 for k in candidate_labels},
    }

    all_texts: List[str] = []
    all_refs: List[Tuple[str, int, Dict[str, Any]]] = []

    # First pass: load videos & comments, prepare per-video result containers and collect texts
    for video_json in _iter_channel_comment_files(channel_dir):
        video_id, comments = _load_comments_from_video_json(video_json)
        video_result = {
            "comments": [],
            "topic_distribution": {k: 0.0 for k in candidate_labels},
            # store the source path so saving can merge with original metadata
            "_source_path": str(video_json),
        }

        if not comments:
            results["videos"][video_id] = video_result
            continue

        for idx, c in enumerate(comments):
            text = _clean_comment_text(c.get("comment", ""))
            if not text:
                c["assigned_topics"] = []
                video_result["comments"].append(c)
            else:
                # placeholder to keep order; will fill after classification
                video_result["comments"].append(None)
                all_texts.append(text)
                all_refs.append((video_id, idx, c))

        results["videos"][video_id] = video_result

    if not all_texts:
        return results

    pipe_out = classifier(
        all_texts,
        candidate_labels_aug,
        hypothesis_template=hypothesis_template,
        multi_label=multi_label,
        batch_size=batch_size,
    )
    if isinstance(pipe_out, dict):
        pipe_out = [pipe_out]

    for i, single_out in enumerate(pipe_out):
        video_id, pos_idx, comment_obj = all_refs[i]
        labels = single_out["labels"]
        scores = single_out["scores"]

        if augment_label_with_definition:
            labels_clean = [l.split(" — ", 1)[0] for l in labels]
        else:
            labels_clean = labels

        label_scores = list(zip(labels_clean, scores))

        accepted: List[Tuple[str, float]] = []
        if multi_label:
            for label, score in label_scores:
                if score >= score_threshold:
                    accepted.append((label, float(score)))
            if top_k is not None and len(accepted) > top_k:
                accepted = sorted(accepted, key=lambda x: x[1], reverse=True)[:top_k]
        else:
            top_label, top_score = label_scores[0]
            if top_score >= score_threshold:
                accepted = [(top_label, float(top_score))]
            else:
                accepted = []

        if not accepted:
            top_label, top_score = label_scores[0]
            if top_score < (score_threshold * 0.6):
                accepted = [("Off-Topic & Unrelated", float(top_score))]

        comment_obj["assigned_topics"] = [
            {"label": label, "score": score} for label, score in accepted
        ]

        # update distributions and insert comment back into its video comments list
        for label, score in accepted:
            canonical_label = label
            if canonical_label not in results["videos"][video_id]["topic_distribution"]:
                results["videos"][video_id]["topic_distribution"].setdefault(
                    canonical_label, 0.0
                )
                results["global_topic_distribution"].setdefault(canonical_label, 0.0)
            results["videos"][video_id]["topic_distribution"][canonical_label] += score
            results["global_topic_distribution"][canonical_label] = (
                results["global_topic_distribution"].get(canonical_label, 0.0) + score
            )

        results["videos"][video_id]["comments"][pos_idx] = comment_obj

    # Normalize distributions to percentages (0.0 - 1.0)
    for v_res in results["videos"].values():
        dist = v_res["topic_distribution"]
        total_score = sum(dist.values())
        if total_score > 0:
            for k in dist:
                dist[k] /= total_score

    # Also normalize global distribution
    g_dist = results["global_topic_distribution"]
    g_total = sum(g_dist.values())
    if g_total > 0:
        for k in g_dist:
            g_dist[k] /= g_total

    return results


def save_results_to_files(
    results: Dict[str, Any],
    data_root: str = "data",
    overwrite: bool = False,
    output_suffix: str = "",
):
    """
    Saves classification results back into video JSONs while preserving original metadata.

    Args:
        results: output from zero_shot_classify_channel.
        data_root: root path containing channel directories.
        overwrite: if True, overwrite the original video JSON file; otherwise write a new file
                   with `output_suffix` appended before the .json extension.
        output_suffix: suffix to append to filename when not overwriting (default: '', i.e. overwriting).
    """
    channel_id = results.get("channel_id")
    if not channel_id:
        print("Error: 'channel_id' not found in results.")
        return

    channel_dir = Path(data_root) / channel_id
    if not channel_dir.exists() or not channel_dir.is_dir():
        print(f"Error: Channel directory not found: {channel_dir}")
        return

    # Update channel_metadata.json with global topic distribution
    metadata_path = channel_dir / "channel_metadata.json"
    try:
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        metadata["global_topic_distribution"] = results.get(
            "global_topic_distribution", {}
        )

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        print(
            f"Updated channel metadata with global topic distribution: {metadata_path}"
        )

    except Exception as e:
        print(f"Error updating channel metadata: {e}")

    # Save per-video classification results while preserving the original video JSON metadata
    for video_id, video_data in results.get("videos", {}).items():
        source_path_str = video_data.get("_source_path")
        source_path = (
            Path(source_path_str)
            if source_path_str
            else channel_dir / f"{video_id}.json"
        )

        # fallback: try to find a matching json file if explicit _source_path is missing
        if not source_path.exists():
            # attempt to find a file that starts with video_id
            matches = list(channel_dir.glob(f"{video_id}*.json"))
            if matches:
                source_path = matches[0]

        output_path = None
        try:
            if source_path.exists():
                # load original metadata
                with open(source_path, "r", encoding="utf-8") as f:
                    original_data = json.load(f)
                # Replace comments with classified comments (preserving all other keys)
                original_data["comments"] = video_data.get(
                    "comments", original_data.get("comments", [])
                )

                if overwrite:
                    output_path = source_path
                else:
                    # create a new filename adding the suffix before .json
                    stem = source_path.stem
                    new_name = f"{stem}{output_suffix}.json"
                    output_path = source_path.parent / new_name

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(original_data, f, indent=4)

                print(f"Saved classified results for video {video_id} to {output_path}")

            else:
                # If the original file wasn't found, write a standalone file containing
                # the comments and topic_distribution (but warn the user).
                fallback = {
                    "video_id": video_id,
                    "comments": video_data.get("comments", []),
                    "topic_distribution": video_data.get("topic_distribution", {}),
                    "_note": "Original source file not found; this file contains only classification results.",
                }
                out_name = f"{video_id}{output_suffix}.json"
                output_path = channel_dir / out_name
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(fallback, f, indent=4)
                print(
                    f"Original source for {video_id} not found. Wrote fallback to {output_path}"
                )

        except Exception as e:
            print(f"Error saving results for video {video_id}: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Zero-shot classify YouTube comments in a channel and optionally save results back to files."
    )
    parser.add_argument(
        "--channel_id",
        help="Channel directory name under data_root to process (required).",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root folder containing channel directories (default: %(default)s).",
    )
    parser.add_argument(
        "--model-name",
        default="joeddav/xlm-roberta-large-xnli",
        help="HuggingFace model name for zero-shot NLI (default: %(default)s).",
    )
    parser.add_argument(
        "--hypothesis-template",
        default="This YouTube comment discusses {}.",
        help="Hypothesis template for zero-shot classification (default: %(default)s).",
    )
    parser.add_argument(
        "--single-label",
        action="store_true",
        help="Use single-label classification mode instead of multi-label (mutually exclusive).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.35,
        help="Minimum score to accept a label (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for classification (default: %(default)s).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Maximum number of labels to keep per comment when multi-label (default: no limit).",
    )
    parser.add_argument(
        "--augment-label-with-definition",
        action="store_true",
        help="Augment candidate labels with their human-readable definitions when classifying.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to files (only run classification).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original video JSON files when saving.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix to append to output filenames when not overwriting.",
    )

    args = parser.parse_args()

    try:
        results = zero_shot_classify_channel(
            channel_id=args.channel_id,
            data_root=args.data_root,
            model_name=args.model_name,
            hypothesis_template=args.hypothesis_template,
            multi_label=(not args.single_label),
            score_threshold=args.score_threshold,
            top_k=args.top_k,
            augment_label_with_definition=args.augment_label_with_definition,
            batch_size=args.batch_size,
        )

        print(f"Classification completed for channel: {args.channel_id}")
        n_videos = len(results.get("videos", {}))
        print(f"Videos processed: {n_videos}")

        if not args.no_save:
            save_results_to_files(
                results,
                data_root=args.data_root,
                overwrite=args.overwrite,
                output_suffix=args.output_suffix,
            )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
