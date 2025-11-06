import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def analyze_channel_sentiment(channel_id, data_root="data", batch_size=64):
    # Define device and model paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # Define label mapping
    label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

    # Prepare channel directory path
    channel_dir = os.path.join(data_root, channel_id)
    if not os.path.exists(channel_dir):
        print(f"Channel directory not found: {channel_dir}")
        return

    # Process each video file in the channel directory
    for filename in tqdm(os.listdir(channel_dir)):
        if filename == "channel_metadata.json" or not filename.endswith(".json"):
            continue

        filepath = os.path.join(channel_dir, filename)

        try:
            # Load video data
            with open(filepath, "r", encoding="utf-8") as f:
                video_data = json.load(f)

            comments = video_data.get("comments", [])
            if not comments:
                continue

            # Extract comment texts
            comment_texts = [c["comment"] for c in comments]
            total_comments = len(comment_texts)

            # Process in batches
            for i in range(0, total_comments, batch_size):
                batch_texts = comment_texts[i : i + batch_size]

                # Tokenize and move to device
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)

                # Predict sentiment probabilities
                with torch.no_grad():
                    outputs = model(**inputs)

                # Calculate probabilities
                probs = (
                    torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
                )

                # Update comments with sentiment probabilities
                for j, prob in enumerate(probs):
                    idx = i + j
                    sentiment_dict = {
                        label_mapping[k]: round(float(v), 2) for k, v in enumerate(prob)
                    }
                    comments[idx]["sentiment"] = sentiment_dict

            # Save updated data
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(video_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze sentiment of YouTube comments"
    )

    parser.add_argument("channel_id", help="The YouTube channel ID to analyze")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory for data files (default: data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for processing comments (default: 64)",
    )

    args = parser.parse_args()

    analyze_channel_sentiment(
        channel_id=args.channel_id, data_root=args.data_root, batch_size=args.batch_size
    )
