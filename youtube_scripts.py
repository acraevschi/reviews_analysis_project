from googleapiclient.discovery import build
import json
import ollama
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import math
import re
import os
from datetime import datetime
from config import API_KEY # replace API_KEY in config.py with your actual API key

def format_date(iso_string):
    try:
        return datetime.strptime(iso_string, "%Y-%m-%dT%H:%M:%S.%fZ").strftime(
            "%Y-%m-%d"
        )
    except:
        # If the first format fails, try the format without microseconds
        try:
            return datetime.strptime(iso_string, "%Y-%m-%dT%H:%M:%SZ").strftime(
                "%Y-%m-%d"
            )
        except:
            return iso_string

def get_channel_info(channel_identifier):
    """
    Fetches channel metadata using a username, direct channel URL, custom URL, or handle (@username).
    """
    youtube = build("youtube", "v3", developerKey=API_KEY)

    # Check if input is a YouTube URL
    if "youtube.com" in channel_identifier:
        # Handle different YouTube URL formats
        match = re.search(r"youtube\.com\/(channel|c|@)\/([^\/?]+)", channel_identifier)
        if match:
            identifier_type = match.group(1)  # "channel", "c", or "@"
            identifier_value = match.group(2)

            if identifier_type == "channel":
                # Direct channel ID, we can use it immediately
                channel_id = identifier_value
            else:
                # Custom URL (/c/) or handle (@), resolve via search API
                search_response = (
                    youtube.search()
                    .list(
                        part="snippet", q=identifier_value, type="channel", maxResults=1
                    )
                    .execute()
                )

                if "items" in search_response and search_response["items"]:
                    channel_id = search_response["items"][0]["snippet"]["channelId"]
                else:
                    print("Channel not found!")
                    return None
        else:
            # Special case: Handle URLs (e.g., youtube.com/@IGN)
            match_handle = re.search(r"youtube\.com\/@([^\/?]+)", channel_identifier)
            if match_handle:
                handle_name = match_handle.group(1)
                search_response = (
                    youtube.search()
                    .list(part="snippet", q=handle_name, type="channel", maxResults=1)
                    .execute()
                )

                if "items" in search_response and search_response["items"]:
                    channel_id = search_response["items"][0]["snippet"]["channelId"]
                else:
                    print("Channel not found!")
                    return None
            else:
                print("Invalid YouTube channel URL!") 
                return None
    else:
        # If it's not a URL, assume it's a username
        response = (
            youtube.channels()
            .list(part="snippet,contentDetails", forUsername=channel_identifier)
            .execute()
        )

        if "items" in response and response["items"]:
            channel_id = response["items"][0]["id"]
        else:
            print("Username not found, try using a direct channel link.")
            return None

    # Now fetch full channel details using the correct channel ID
    channel_response = (
        youtube.channels()
        .list(part="snippet,contentDetails,statistics", id=channel_id)
        .execute()
    )

    if "items" not in channel_response or not channel_response["items"]:
        print("Channel ID lookup failed.")
        return None

    channel_data = channel_response["items"][0]
    return {
        "channel_id": channel_id,
        "username": channel_data["snippet"]["title"],
        "description": channel_data["snippet"]["description"],
        "creation_date": format_date(channel_data["snippet"]["publishedAt"]),
        "uploads_playlist_id": channel_data["contentDetails"]["relatedPlaylists"][
            "uploads"
        ],
        "view_count": channel_data["statistics"]["viewCount"],
        "subscriber_count": channel_data["statistics"]["subscriberCount"],
        "video_count": channel_data["statistics"]["videoCount"],
    }


def get_last_videos(playlist_id, N=10):
    """
    Retrieves the last N video IDs from the given playlist.
    """
    youtube = build("youtube", "v3", developerKey=API_KEY)
    request = youtube.playlistItems().list(
        part="snippet", playlistId=playlist_id, maxResults=N
    )
    response = request.execute()

    videos = []
    for item in response.get("items", []):
        video_id = item["snippet"]["resourceId"]["videoId"]
        title = item["snippet"]["title"]
        published_at = format_date(item["snippet"]["publishedAt"])

        videos.append(
            {"video_id": video_id, "title": title, "published_at": published_at}
        )

    return videos


def get_video_metadata(video_ids):
    """
    Fetches metadata for a list of videos, including view count, like count, comment count,
    publication date, and title.
    """
    if type(video_ids) is str:
        video_ids = [video_ids]  # Ensure video_ids is a list
        
    video_metadata = {}

    youtube = build("youtube", "v3", developerKey=API_KEY)
    request = youtube.videos().list(part="snippet,statistics", id=",".join(video_ids))

    try:
        response = request.execute()
        for item in response.get("items", []):
            video_id = item["id"]
            snippet = item["snippet"]
            statistics = item.get("statistics", {})
            video_metadata[video_id] = {
                "title": snippet["title"],
                "description": snippet.get("description", ""),
                "published_at": format_date(snippet["publishedAt"]),
                "view_count": statistics.get("viewCount", "0"),
                "like_count": statistics.get("likeCount", "0"),
                "comment_count": statistics.get("commentCount", "0"),
            }
    except Exception as e:
        print(f"Error fetching video metadata: {e}")

    return video_metadata


def get_video_transcripts(video_ids, ollama_model=None):
    """
    Retrieves transcripts for a list of video IDs using youtube_transcript_api.
    Attempts to fetch in multiple languages if available.
    If ollama_model is provided, summarizes the transcript using Ollama.
    """
    languages = [
        "en", "fr", "de", "ru", "ar", "zh", "hi", "ur", "tr", "es", "it", "id",
        "pt", "ja", "ko", "nl", "sv", "pl", "th", "vi",
    ]
    
    if ollama_model:
        try:
            ollama.pull(ollama_model)
            check_response = ollama.generate(model=ollama_model, prompt="That's a test, are you working? Answer just with 'yes' or 'no'.")
            if check_response.get("done"):
                print(f"'{ollama_model}' is ready to summarize.")
            else:
                print(f"'{ollama_model}' is not working.")
        except Exception as e:
            print(f"Ollama model creation error: {e}\n Using Ollama model is disabled.")
            ollama_model = None

    video_transcripts = {}
    summary_transcripts = {}

    for vid in video_ids:
        try:
            transcript_instance = YouTubeTranscriptApi()
            transcript = transcript_instance.fetch(video_id=vid, languages=languages)
            transcript_text = " ".join([t.text for t in transcript])
            video_transcripts[vid] = transcript_text

            if ollama_model and transcript_text.strip():
                try:
                    prompt = (
                        "Summarize the following YouTube video transcript in no more than 250 words."
                        "Return ONLY the summary, and wrap it in <summary>...</summary> XML tags. "
                        "Do not add any comments or explanations before or after the summary.\n\n"
                        f"{transcript_text}"
                    )
                    response = ollama.chat(model=ollama_model, messages=[{"role": "user", "content": prompt}])
                    summary = response["message"]["content"] if "message" in response else ""

                    summary_transcripts[vid] = summary
                except Exception as e:
                    print(f"Error summarizing transcript for {vid}: {e}")
                    summary_transcripts[vid] = ""
        except TranscriptsDisabled:
            print(f"Transcripts are disabled for video {vid}.")
            video_transcripts[vid] = ""
            summary_transcripts[vid] = ""
        except Exception as e:
            print(f"Error fetching transcript for {vid}: {e}")
            video_transcripts[vid] = ""
            summary_transcripts[vid] = ""

    return video_transcripts, summary_transcripts


def get_comments(video_id, num_comments=100):
    """
    Retrieves top-level comments for a given video.
    """
    youtube = build("youtube", "v3", developerKey=API_KEY)

    num_to_request = min(num_comments, 100)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=num_to_request,
    )

    n_requests = math.ceil(num_comments / 100)
    comments = []

    for _ in range(n_requests):
        try:
            response = request.execute()
            if "items" not in response:  # Ensure there are comments
                break
        except Exception as e:
            print(f"Error fetching comments for video {video_id}: {e}")
            break

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(
                {
                    "comment_id": item["id"],
                    "video_id": video_id,
                    "author": snippet["authorDisplayName"],
                    "date": format_date(snippet["publishedAt"]),
                    "likes": snippet["likeCount"],
                    "comment": snippet["textDisplay"],
                    "num_replies": item["snippet"]["totalReplyCount"],
                }
            )

        # Get the next page of comments (if any)
        request = youtube.commentThreads().list_next(request, response)
        if request is None:
            break

    return comments


def fetch_channel_data(
    channel_url,
    num_videos=10,
    num_comments=50,
    data_dir="./data/",
    ollama_model=None
):
    """
    Fetches channel metadata, latest videos, transcripts, video metadata, and comments,
    then writes the data into separate JSON files in a structured directory.
    """
    channel_info = get_channel_info(channel_url)
    if not channel_info:
        print("Channel not found!")
        return None

    channel_folder = os.path.join(data_dir, channel_info["channel_id"])
    os.makedirs(channel_folder, exist_ok=True)

    channel_metadata_filename = os.path.join(channel_folder, "channel_metadata.json")
    with open(channel_metadata_filename, "w", encoding="utf-8") as f:
        json.dump(channel_info, f, indent=4, ensure_ascii=False)

    playlist_id = channel_info["uploads_playlist_id"]
    videos = get_last_videos(playlist_id, num_videos)
    video_ids = [video["video_id"] for video in videos]
    transcripts, summary_transcripts = get_video_transcripts(
        video_ids, ollama_model=ollama_model
    )
    comments_data = {vid: get_comments(vid, num_comments) for vid in video_ids}
    video_metadata_dict = get_video_metadata(video_ids)

    for video in videos:
        vid = video["video_id"]
        transcript_text = transcripts.get(vid, "")
        summary_text = summary_transcripts.get(vid, "")
        video_metadata = video_metadata_dict.get(vid, {})

        # Merge all video_metadata_dict items at the top level
        video_data = {
            **video_metadata,
            "transcript": transcript_text,
            "summary": summary_text,
            "comments": comments_data.get(vid, []),
        }
        video_filename = os.path.join(channel_folder, f"{vid}.json")
        with open(video_filename, "w", encoding="utf-8") as vf:
            json.dump(video_data, vf, indent=4, ensure_ascii=False)

    print(f"Data saved to folder: {channel_folder}")
    return channel_info

if __name__ == "__main__":
    channel_url = input("Enter YouTube channel URL or identifier: ")
    num_videos = int(input("Enter number of latest videos to fetch: "))
    num_comments = int(input("Enter number of comments to fetch per video: "))
    data_dir = input("Enter directory to save data (default: ./data/): ") or "./data/"
    ollama_model = input("Enter Ollama model name (or leave blank for no summarization): ") or None
    fetch_channel_data(
        channel_url,
        num_videos=num_videos,
        num_comments=num_comments,
        data_dir=data_dir,
        ollama_model=ollama_model
    )
