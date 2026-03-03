import os
import json
import random
import re
from googleapiclient.discovery import build
from config import API_KEY

VIDEOS_PER_QUERY = 3
COMMENTS_PER_VIDEO = 10

# Multilingual queries to ensure diverse content types and languages
SEARCH_QUERIES = [
    # English
    "vlog", "tech review", "gaming walkthrough", "cooking tutorial", "podcast episode",  
    # Spanish
    "noticias de hoy", "tutorial de maquillaje", "reseña de telefono", "documental historia", "clases de guitarra",
    # Russian
    "обзор фильма", "влог путешествие", "прохождение игры", "рецепт торта", "уроки программирования",
    # Chinese (Simplified)
    "科技评测", "吃播", "日常vlog", "深度学习教程", "新闻联播",
    # French
    "recette de cuisine", "actualité", "critique de film", "astuces beauté", "vulgarisation scientifique",
    # German
    "let's play deutsch", "produkttest", "nachrichten", "dokumentation", "fitness workout",
    # Portuguese
    "gameplay brasil", "receita fácil", "dicas de maquiagem", "notícias do dia", "curso de inglês",
    # Hindi
    "आज की ताजा खबर", "कुकिंग रेसिपी", "गेमिंग वीडियो", "टेक रिव्यू", "व्लॉग",
    # Arabic
    "بث مباشر العاب", "اخبار اليوم", "مراجعة هواتف", "وصفات طبخ", "فلوق سفر",
    # Japanese
    "ゲーム実況", "料理レシピ", "メイク動画", "ニュース", "日常vlog",
    # Korean
    "브이로그", "게임 플레이", "먹방", "제품 리뷰", "메이크업 튜토리얼"
]

def format_duration(iso_duration: str) -> int:
    """Converts YouTube's ISO 8601 duration format into total seconds."""
    total_seconds = 0
    # Matches patterns like PT1H2M10S, PT5M33S, etc.
    match = re.match(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration)
    if match:
        days, hours, minutes, seconds = [int(x) if x else 0 for x in match.groups()]
        total_seconds = (days * 86400) + (hours * 3600) + (minutes * 60) + seconds
    return total_seconds

def get_random_videos(youtube, query, max_results):
    """Searches YouTube for a query and returns video IDs."""
    print(f"Searching for: {query}")
    try:
        # We use videoCategoryId="10" (Music) or just broad search to get random stuff
        # Using type="video" is required
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results,
            order="relevance" # 'date' or 'rating' also work for OOD
        )
        response = request.execute()
        return [item["id"]["videoId"] for item in response.get("items", [])]
    except Exception as e:
        print(f"Error searching for {query}: {e}")
        return []

def get_video_details_and_comments(youtube, video_id):
    """Fetches video title, description, duration, and samples random comments."""
    try:
        # 1. Get Metadata (Added 'contentDetails' to the 'part' parameter)
        vid_request = youtube.videos().list(part="snippet,contentDetails", id=video_id)
        vid_response = vid_request.execute()
        
        if not vid_response.get("items"):
            return None
            
        item = vid_response["items"][0]
        snippet = item["snippet"]
        content_details = item.get("contentDetails", {})
        
        title = snippet["title"]
        description = snippet.get("description", "")
        channel_id = snippet["channelId"]
        
        # Extract and format the duration
        iso_duration = content_details.get("duration", "PT0S")
        duration_seconds = format_duration(iso_duration)

        # 2. Get Comments
        comments = []
        comment_request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100  # Pull 100 to ensure a good pool for random sampling
        )
        comment_response = comment_request.execute()
        
        for comment_item in comment_response.get("items", []):
            text = comment_item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            # Clean up excessive whitespace but keep language characters
            text = " ".join(text.strip().split())
            if text and len(text) > 2: # Skip completely empty or 1-char comments
                comments.append(text)

        # Sample randomly using our updated constant
        if len(comments) > COMMENTS_PER_VIDEO:
            comments = random.sample(comments, COMMENTS_PER_VIDEO)

        if not comments:
            return None

        return {
            "video_id": video_id,
            "channel_id": channel_id,
            "title": title,
            "description": description,
            "duration_seconds": duration_seconds,  # New field added here
            "comments": comments
        }
    except Exception as e:
        # Comments might be disabled, ignore and move on
        return None

def main():
    youtube = build("youtube", "v3", developerKey=API_KEY)
    dataset = []

    for query in SEARCH_QUERIES:
        video_ids = get_random_videos(youtube, query, VIDEOS_PER_QUERY)
        for vid in video_ids:
            data = get_video_details_and_comments(youtube, vid)
            if data:
                dataset.append(data)

    output_dir = "classification_data"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "raw_samples.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
        
    total_comments = sum(len(v["comments"]) for v in dataset)
    print(f"\nSaved {len(dataset)} videos with a total of {total_comments} comments to {output_path}")

if __name__ == "__main__":
    main()