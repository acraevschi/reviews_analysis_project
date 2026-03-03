import os
import json
import asyncio
import re
from openai import AsyncOpenAI

# The 5 Actionable Creator Categories we defined earlier
FLAGS = ["is_request", "is_question", "is_highlight", "is_feedback", "is_spam"]

SYSTEM_PROMPT = """You are an expert YouTube dataset labeler. Your job is to classify YouTube comments in various languages into binary flags to help creators analyze their audience.

Evaluate the provided Video Title, Video Description, and Comment. Assign a 1 (True) or 0 (False) for each of the following flags. A comment can have multiple flags set to 1.

1. is_request: Asking the creator to make a specific video, cover a topic, or try a product next.
2. is_question: Asking for clarification about the video's content or a related topic; a genuine question.
3. is_highlight: Quoting the video, providing a timestamp, or explicitly praising a specific moment.
4. is_feedback: Actionable critique on production, pacing, audio, editing, or presentation.
5. is_spam: Links, bot behavior, gibberish, or highly toxic/unrelated noise.

CRITICAL INSTRUCTIONS:
- You MUST output a raw JSON dictionary and nothing else.
- Do NOT use markdown formatting (like ```json).
- The JSON must strictly match this exact template:
{
    "is_request": 0,
    "is_question": 0,
    "is_highlight": 0,
    "is_feedback": 0,
    "is_spam": 0
}

Example Output:
{"is_request": 1, "is_question": 0, "is_highlight": 0, "is_feedback": 1, "is_spam": 0}
"""

def parse_reasoning_output(raw_output: str) -> dict:
    """Extracts the JSON payload from a reasoning model's output."""
    # Strip out the <think> block if it exists
    content = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
    
    # Strip potential markdown blocks just in case the model disobeys
    content = re.sub(r'```(?:json)?', '', content).strip()
    
    try:
        # Find the first { and the last }
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = content[start_idx:end_idx+1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
        
    return None

async def label_comment(client: AsyncOpenAI, model: str, title: str, desc: str, comment: str, semaphore: asyncio.Semaphore):
    user_prompt = f"Title: {title}\nDescription: {desc}\nComment: {comment}"
    
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                reasoning_effort="low",       # Use "low" or "medium" as planned
                max_completion_tokens=2048    # Accounts for reasoning budget + final output
                # Note: temperature is deliberately omitted for gpt-oss
            )
            
            # LM Studio outputs the final JSON here. 
            # (If you want to log the reasoning for auditing, it is in response.choices[0].message.reasoning)
            raw_output = response.choices[0].message.content
            
            parsed_json = parse_reasoning_output(raw_output)
            
            if parsed_json and all(k in parsed_json for k in FLAGS):
                return {"comment": comment, "labels": parsed_json}
            else:
                print(f"Failed to parse or invalid keys: {raw_output.strip()}")
                return None
                
        except Exception as e:
            print(f"API Error: {e}")
            return None

async def main():
    api_base = "http://localhost:1234/v1"
    model_name = "reasoning-model" # LM Studio usually ignores this, but it's good practice
    max_concurrent = 4
    
    client = AsyncOpenAI(base_url=api_base, api_key="local")
    semaphore = asyncio.Semaphore(max_concurrent)
    
    input_path = os.path.join("classification_data", "raw_samples.json")
    output_path = os.path.join("classification_data", "labeled_dataset.json")
    
    if not os.path.exists(input_path):
        print(f"Could not find {input_path}. Run sample_videos_comments.py first.")
        return
        
    with open(input_path, "r", encoding="utf-8") as f:
        videos = json.load(f)
        
    labeled_data = []
    
    # count = 0 # temporary
    for video in videos:
        # if count == 5:
        #     break
        print(f"Processing video: {video['title']}")
        tasks = []
        for comment in video['comments']:
            description = video["description"]
            description_trimmed = description[:500] + "..." if len(description) > 500 else description
            tasks.append(label_comment(client, model_name, video['title'], description_trimmed, comment, semaphore))
            
        results = await asyncio.gather(*tasks)
        
        # Filter out failed parses
        valid_results = [r for r in results if r is not None]
        
        if valid_results:
            labeled_data.extend([{
                "video_id": video["video_id"],
                "video_title": video["title"],
                "video_description": video["description"],
                "comment": r["comment"],
                **r["labels"] # Flatten the labels into the main dict
            } for r in valid_results])
        # count += 1
        # continue
        

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labeled_data, f, indent=4, ensure_ascii=False)
        
    print(f"\nSuccessfully labeled {len(labeled_data)} comments. Saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())