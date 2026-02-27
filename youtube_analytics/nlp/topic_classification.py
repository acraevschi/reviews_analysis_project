import json
import argparse
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from openai import AsyncOpenAI

CATEGORY_DEFINITIONS = {
    "Requests & Ideas": "Suggestions for future videos, topics, or things viewers want to see next.",
    "Questions & Confusion": "Viewers asking for clarification, more details, or expressing they didn't understand something.",
    "Highlights & Quotes": "Praise for specific moments, timestamps, or quotes from the video.",
    "Constructive Criticism": "Actionable feedback on how to improve the content, pacing, structure, or delivery of the creator.",
    "Production Quality": "Feedback on audio, lighting, editing, camera work, or technical issues like sync problems.",
    "Praise & Appreciation": "General positive feedback, expressing gratitude, or complimenting the creator.",
    "Community & Stories": "Viewers sharing personal anecdotes, relating to the topic, or having discussions with each other.",
    "Spam & Noise": "Irrelevant, promotional, nonsensical, or completely off-topic comments.",
}

# Construct the schema for the prompt
CATEGORIES_PROMPT = "\n".join([f"- **{k}**: {v}" for k, v in CATEGORY_DEFINITIONS.items()])

def truncate_description(desc: str, max_words: int = 250) -> str:
    """Smartly truncates the description so it doesn't flood the context window."""
    if not desc:
        return ""
    words = desc.split()
    if len(words) <= max_words:
        return desc
    return " ".join(words[:max_words]) + "..."

async def classify_comment_llm(
    client: AsyncOpenAI, 
    model: str, 
    title: str, 
    description: str, 
    comment: str, 
    semaphore: asyncio.Semaphore
) -> List[str]:
    """Asynchronously calls the local LLM to classify a single comment."""
    
    system_prompt = f"""You are a highly accurate YouTube comment classifier.
Analyze the provided user comment based on the context of the video's title and description.
Assign the comment to 1 to 3 of the following categories:
{CATEGORIES_PROMPT}

CRITICAL INSTRUCTIONS:
1. You MUST respond ONLY with a valid JSON array of strings.
2. The strings MUST perfectly match the category names listed above.
3. Do not provide any conversational text, markdown formatting blocks (like ```json), or explanations.
Example output:
["Video Content", "Personality"]
"""

    user_prompt = f"""Video Title: {title}
Video Description: {description}
Comment: {comment}"""

    # Use semaphore to limit concurrent connections to the local server
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )
            
            output = response.choices[0].message.content.strip()
            
            # Clean up potential markdown wrappers
            if output.startswith("```json"):
                output = output[7:]
            if output.startswith("```"):
                output = output[3:]
            if output.endswith("```"):
                output = output[:-3]
                
            categories = json.loads(output.strip())
            
            # Validation
            if not isinstance(categories, list):
                categories = [categories]
                
            # Filter strictly to defined keys
            valid_categories = [c for c in categories if c in CATEGORY_DEFINITIONS]
            
            # Enforce max 3 and fallback
            valid_categories = valid_categories[:3]
            if not valid_categories:
                return ["Off-Topic & Unrelated"]
                
            return valid_categories

        except Exception as e:
            # Fallback on parsing error or server timeout
            print(f"Error classifying comment: {e}")
            return ["Off-Topic & Unrelated"]

async def process_video_comments(
    client: AsyncOpenAI, 
    model: str, 
    video_data: Dict[str, Any], 
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    
    title = video_data.get("title", "Unknown Title")
    description = truncate_description(video_data.get("description", ""))
    comments = video_data.get("comments", [])
    
    if not comments:
        return video_data

    tasks = []
    for c in comments:
        text = " ".join(c.get("comment", "").strip().split())
        if not text:
            tasks.append(asyncio.sleep(0, result=["Off-Topic & Unrelated"])) # Fake task for empty comments
        else:
            tasks.append(classify_comment_llm(client, model, title, description, text, semaphore))
            
    # Gather results concurrently
    classification_results = await asyncio.gather(*tasks)
    
    topic_distribution = {k: 0.0 for k in CATEGORY_DEFINITIONS}
    
    for c, assigned_labels in zip(comments, classification_results):
        # Evenly distribute a score of 1.0 across the assigned labels 
        # (e.g. 2 labels = 0.5 score each)
        score_per_label = 1.0 / len(assigned_labels)
        c["assigned_topics"] = [{"label": label, "score": score_per_label} for label in assigned_labels]
        
        for label in assigned_labels:
            topic_distribution[label] += score_per_label

    # Normalize distribution
    total_score = sum(topic_distribution.values())
    if total_score > 0:
        for k in topic_distribution:
            topic_distribution[k] /= total_score

    video_data["topic_distribution"] = topic_distribution
    return video_data

async def run_classification_pipeline(
    channel_id: str,
    data_root: str,
    api_base: str,
    model_name: str,
    max_concurrent: int,
    overwrite: bool,
    output_suffix: str
):
    channel_dir = Path(data_root) / channel_id
    if not channel_dir.exists() or not channel_dir.is_dir():
        raise FileNotFoundError(f"Channel directory not found: {channel_dir}")

    # Set up OpenAI Client
    # Uses a dummy API key for local servers like LM Studio or vLLM
    client = AsyncOpenAI(base_url=api_base, api_key="local-server")
    semaphore = asyncio.Semaphore(max_concurrent)

    global_distribution = {k: 0.0 for k in CATEGORY_DEFINITIONS}
    video_files = [p for p in channel_dir.glob("*.json") if p.name != "channel_metadata.json"]
    
    for video_json_path in video_files:
        with open(video_json_path, "r", encoding="utf-8") as f:
            video_data = json.load(f)
            
        print(f"Processing video: {video_data.get('title', video_json_path.stem)}")
        processed_video_data = await process_video_comments(client, model_name, video_data, semaphore)
        
        # Accumulate to global distribution
        vd_dist = processed_video_data.get("topic_distribution", {})
        for k, v in vd_dist.items():
            global_distribution[k] += v

        # Save back to file
        output_path = video_json_path if overwrite else video_json_path.parent / f"{video_json_path.stem}{output_suffix}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_video_data, f, indent=4)
            
        print(f"Saved results to {output_path}")

    # Normalize Global Distribution
    total_global = sum(global_distribution.values())
    if total_global > 0:
        for k in global_distribution:
            global_distribution[k] /= total_global

    # Update metadata
    metadata_path = channel_dir / "channel_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        metadata["global_topic_distribution"] = global_distribution
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        print("Updated channel metadata with global topic distribution.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-based YouTube comments classification.")
    parser.add_argument("--channel_id", required=True, help="Channel directory name under data_root.")
    parser.add_argument("--data-root", default="data", help="Root folder containing channel directories.")
    parser.add_argument("--api-base", default="http://localhost:1234/v1", help="URL of the local inference server (LM Studio defaults to http://localhost:1234/v1).")
    parser.add_argument("--model-name", default="local-model", help="Name of the model (Often ignored by LM Studio, but required by API spec).")
    parser.add_argument("--max-concurrent", type=int, default=8, help="Number of concurrent requests to send to the local server.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite original video JSON files.")
    parser.add_argument("--output-suffix", default="_classified", help="Suffix to append to output filenames if not overwriting.")

    args = parser.parse_args()

    # Run the asyncio event loop
    try:
        asyncio.run(run_classification_pipeline(
            channel_id=args.channel_id,
            data_root=args.data_root,
            api_base=args.api_base,
            model_name=args.model_name,
            max_concurrent=args.max_concurrent,
            overwrite=args.overwrite,
            output_suffix=args.output_suffix
        ))
        print(f"Process completed successfully.")
    except Exception as e:
        print(f"Error during execution: {e}")
        sys.exit(1)