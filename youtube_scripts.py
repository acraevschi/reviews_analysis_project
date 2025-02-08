from googleapiclient.discovery import build
import json
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import math
import re

# Replace with your API key
API_KEY = "AIzaSyA0nnuhV-1yVMRAEAX7yOCHR69cPqK6A1E"

# Initialize YouTube API
youtube = build("youtube", "v3", developerKey=API_KEY)


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
        # If it's not a URL, assume it's a username (legacy)
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
        youtube.channels().list(part="snippet,contentDetails", id=channel_id).execute()
    )

    if "items" not in channel_response or not channel_response["items"]:
        print("Channel ID lookup failed.")
        return None

    channel_data = channel_response["items"][0]

    return {
        "channel_id": channel_id,
        "username": channel_data["snippet"]["title"],
        "description": channel_data["snippet"]["description"],
        "uploads_playlist_id": channel_data["contentDetails"]["relatedPlaylists"][
            "uploads"
        ],
    }


def get_last_videos(playlist_id, N=10):
    """
    Retrieves the last N video IDs from the given playlist.
    """
    request = youtube.playlistItems().list(
        part="snippet", playlistId=playlist_id, maxResults=N
    )
    response = request.execute()

    videos = []
    for item in response.get("items", []):
        video_id = item["snippet"]["resourceId"]["videoId"]
        title = item["snippet"]["title"]
        published_at = item["snippet"]["publishedAt"]

        videos.append(
            {"video_id": video_id, "title": title, "published_at": published_at}
        )

    return videos


def get_video_transcripts(video_ids):
    """
    Retrieves transcripts for a list of video IDs using youtube_transcript_api.
    If an error occurs for a specific video, it returns an empty transcript for that video.
    """
    video_transcripts = {}
    for vid in video_ids:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(vid)
            video_transcripts[vid] = " ".join([t["text"] for t in transcript])
        except Exception as e:
            print(f"Error fetching transcript for {vid}: {e}")
            video_transcripts[vid] = ""  # Empty transcript if failed

    return video_transcripts


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
                    "date": snippet["publishedAt"],
                    "likes": snippet["likeCount"],
                    "comment": snippet["textDisplay"],
                }
            )

        # Ensure request is valid before proceeding
        request = youtube.commentThreads().list_next(request, response)
        if request is None:
            break

    return comments


def fetch_channel_data(channel_url, num_videos=10, num_comments=50):
    """
    Fetches channel metadata, latest videos, transcripts, and comments, then stores everything in JSON.
    """
    # Step 1: Get channel info
    channel_info = get_channel_info(channel_url)
    if not channel_info:
        print("Channel not found!")
        return None

    # Step 2: Get last N videos
    playlist_id = channel_info["uploads_playlist_id"]
    videos = get_last_videos(playlist_id, num_videos)

    # Step 3: Get transcripts
    video_ids = [video["video_id"] for video in videos]
    transcripts = get_video_transcripts(video_ids)

    # Step 4: Get comments
    comments_data = {vid: get_comments(vid, num_comments) for vid in video_ids}

    # Step 5: Structure data in the new format
    structured_data = {
        "channel": {
            "channel_id": channel_info["channel_id"],
            "username": channel_info["username"],
            "description": channel_info["description"],
            "uploads_playlist_id": playlist_id,
            "videos": [
                {
                    "video_id": video["video_id"],
                    "title": video["title"],
                    "published_at": video["published_at"],
                    "transcript": transcripts.get(
                        video["video_id"], ""
                    ),  # Use empty transcript if missing
                    "comments": comments_data.get(video["video_id"], []),
                }
                for video in videos
            ],
        }
    }

    # Generate a safe filename from the channel URL
    safe_filename = re.sub(
        r"\W+", "_", channel_info["username"]
    )  # Replace non-word chars
    json_filename = f"{safe_filename}_youtube_data.json"

    # Save as JSON
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=4)

    print(f"Data saved to {json_filename}")
    return structured_data
