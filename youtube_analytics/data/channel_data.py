from googleapiclient.discovery import build
import json
import math
import re
import os
from datetime import datetime
from config import API_KEY

def format_date(iso_string):
    try:
        return datetime.strptime(iso_string, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d")
    except ValueError:
        try:
            return datetime.strptime(iso_string, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
        except ValueError:
            return iso_string

def format_duration(iso_duration: str) -> int:
    total_seconds = 0
    match = re.match(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration)
    if match:
        days, hours, minutes, seconds = [int(x) if x else 0 for x in match.groups()]
        total_seconds = (days * 86400) + (hours * 3600) + (minutes * 60) + seconds
    return total_seconds

def get_channel_info(channel_identifier):
    youtube = build("youtube", "v3", developerKey=API_KEY)

    if "youtube.com" in channel_identifier:
        match = re.search(r"youtube\.com\/(channel|c|@)\/([^\/?]+)", channel_identifier)
        if match:
            identifier_type = match.group(1)
            identifier_value = match.group(2)

            if identifier_type == "channel":
                channel_id = identifier_value
            else:
                search_response = youtube.search().list(part="snippet", q=identifier_value, type="channel", maxResults=1).execute()
                if "items" in search_response and search_response["items"]:
                    channel_id = search_response["items"][0]["snippet"]["channelId"]
                else:
                    return None
        else:
            match_handle = re.search(r"youtube\.com\/@([^\/?]+)", channel_identifier)
            if match_handle:
                search_response = youtube.search().list(part="snippet", q=match_handle.group(1), type="channel", maxResults=1).execute()
                if "items" in search_response and search_response["items"]:
                    channel_id = search_response["items"][0]["snippet"]["channelId"]
                else:
                    return None
            else:
                return None
    else:
        response = youtube.channels().list(part="snippet,contentDetails", forUsername=channel_identifier).execute()
        if "items" in response and response["items"]:
            channel_id = response["items"][0]["id"]
        else:
            return None

    channel_response = youtube.channels().list(part="snippet,contentDetails,statistics", id=channel_id).execute()
    if not channel_response.get("items"):
        return None

    channel_data = channel_response["items"][0]
    return {
        "channel_id": channel_id,
        "username": channel_data["snippet"]["title"],
        "description": channel_data["snippet"]["description"],
        "creation_date": format_date(channel_data["snippet"]["publishedAt"]),
        "uploads_playlist_id": channel_data["contentDetails"]["relatedPlaylists"]["uploads"],
        "view_count": channel_data["statistics"]["viewCount"],
        "subscriber_count": channel_data["statistics"]["subscriberCount"],
        "video_count": channel_data["statistics"]["videoCount"],
    }

def get_last_videos(playlist_id, N=10):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    request = youtube.playlistItems().list(part="snippet", playlistId=playlist_id, maxResults=N)
    response = request.execute()

    videos = []
    for item in response.get("items", []):
        videos.append({
            "video_id": item["snippet"]["resourceId"]["videoId"],
            "title": item["snippet"]["title"],
            "published_at": format_date(item["snippet"]["publishedAt"])
        })
    return videos

def get_video_metadata(video_ids):
    if isinstance(video_ids, str):
        video_ids = [video_ids]

    video_metadata = {}
    youtube = build("youtube", "v3", developerKey=API_KEY)
    request = youtube.videos().list(part="snippet,statistics,contentDetails,topicDetails", id=",".join(video_ids))

    try:
        response = request.execute()
        for item in response.get("items", []):
            snippet = item["snippet"]
            statistics = item.get("statistics", {})
            content = item.get("contentDetails", {})
            topics = item.get("topicDetails", {})
            video_metadata[item["id"]] = {
                "title": snippet["title"],
                "description": snippet.get("description", ""),
                "published_at": format_date(snippet["publishedAt"]),
                "view_count": statistics.get("viewCount", "0"),
                "like_count": statistics.get("likeCount", "0"),
                "comment_count": statistics.get("commentCount", "0"),
                "duration_seconds": format_duration(content.get("duration", "PT0S")),
                "definition": content.get("definition", ""),
                "caption": content.get("caption", ""),
                "topics": topics.get("topicCategories", []),
            }
    except Exception as e:
        print(f"Error fetching video metadata: {e}")

    return video_metadata

def get_comments(video_id, num_comments=100):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    num_to_request = min(num_comments, 100)
    request = youtube.commentThreads().list(part="snippet", videoId=video_id, textFormat="plainText", maxResults=num_to_request)

    n_requests = math.ceil(num_comments / 100)
    comments = []

    for _ in range(n_requests):
        try:
            response = request.execute()
            if "items" not in response:
                break
        except Exception as e:
            print(f"Error fetching comments for video {video_id}: {e}")
            break

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["id"],
                "video_id": video_id,
                "author": snippet["authorDisplayName"],
                "date": format_date(snippet["publishedAt"]),
                "likes": snippet["likeCount"],
                "comment": snippet["textDisplay"],
                "num_replies": item["snippet"]["totalReplyCount"],
            })

        request = youtube.commentThreads().list_next(request, response)
        if request is None:
            break

    return comments

def fetch_channel_data(channel_url, num_videos=10, num_comments=50, data_dir="./data/"):
    channel_info = get_channel_info(channel_url)
    if not channel_info:
        print("Channel not found!")
        return None

    channel_folder = os.path.join(data_dir, channel_info["channel_id"])
    os.makedirs(channel_folder, exist_ok=True)

    with open(os.path.join(channel_folder, "channel_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(channel_info, f, indent=4, ensure_ascii=False)

    videos = get_last_videos(channel_info["uploads_playlist_id"], num_videos)
    video_ids = [video["video_id"] for video in videos]
    
    comments_data = {vid: get_comments(vid, num_comments) for vid in video_ids}
    video_metadata_dict = get_video_metadata(video_ids)

    for video in videos:
        vid = video["video_id"]
        video_data = {
            **video_metadata_dict.get(vid, {}),
            "comments": comments_data.get(vid, []),
        }
        with open(os.path.join(channel_folder, f"{vid}.json"), "w", encoding="utf-8") as vf:
            json.dump(video_data, vf, indent=4, ensure_ascii=False)

    print(f"Data saved to folder: {channel_folder}")
    return channel_info

if __name__ == "__main__":
    channel_url = input("Enter YouTube channel URL or identifier: ")
    num_videos = int(input("Enter number of latest videos to fetch: "))
    num_comments = int(input("Enter number of comments to fetch per video: "))
    data_dir = input("Enter directory to save data (default: ./data/): ") or "./data/"
    fetch_channel_data(channel_url, num_videos=num_videos, num_comments=num_comments, data_dir=data_dir)