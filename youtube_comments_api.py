from googleapiclient.discovery import build
import pandas as pd
from tqdm import tqdm

### https://www.youtube.com/watch?v=VIDEO_ID&lc=COMMENT_ID

KEY = "AIzaSyA0nnuhV-1yVMRAEAX7yOCHR69cPqK6A1E"


def get_comments(video_link, num_comments, get_responses=True):
    if "youtube" in video_link:
        video_id = video_link.split("v=")[1]  # introduce some error handling here
    elif "youtu.be" in video_link:
        link_part = video_link.split("/")[-1]
        video_id = link_part.split("?")[0]

    youtube = build("youtube", "v3", developerKey=KEY)

    request = youtube.commentThreads().list(
        part="snippet,replies", videoId=video_id, textFormat="plainText", maxResults=100
    )

    n_requests = num_comments // 100

    df = pd.DataFrame(
        columns=[
            "comment_id",
            "refers_to",
            "video_id",
            "author",
            "date",
            "likes",
            "num_replies",
            "comment",
        ]
    )

    for _ in tqdm(range(n_requests)):
        # check how to avoid extracting the comments of the author of the video
        try:
            response = request.execute()
        except:
            print("Request failed, stopping further requests here.")

        for item in response["items"]:
            comment_id = item["id"]
            video_id = item["snippet"]["topLevelComment"]["snippet"]["videoId"]
            author = item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
            date = item["snippet"]["topLevelComment"]["snippet"]["publishedAt"]
            likes = item["snippet"]["topLevelComment"]["snippet"]["likeCount"]
            num_replies = item["snippet"]["totalReplyCount"]
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]

            to_add = pd.DataFrame(
                [
                    {
                        "comment_id": comment_id,
                        "refers_to": pd.NA,
                        "video_id": video_id,
                        "author": author,
                        "date": date,
                        "likes": likes,
                        "num_replies": num_replies,
                        "comment": comment,
                    }
                ]
            )

            df = pd.concat([df, to_add], ignore_index=True)

            if num_replies > 0 and get_responses:

                for reply in item["replies"]["comments"]:
                    reply_comment_id = reply["id"]
                    author = reply["snippet"]["authorDisplayName"]
                    date = reply["snippet"]["publishedAt"]
                    likes = reply["snippet"]["likeCount"]
                    comment = reply["snippet"]["textDisplay"]

                    to_add = pd.DataFrame(
                        [
                            {
                                "comment_id": reply_comment_id,
                                "refers_to": comment_id,
                                "video_id": video_id,
                                "author": author,
                                "date": date,
                                "likes": likes,
                                "num_replies": pd.NA,
                                "comment": comment,
                            }
                        ]
                    )

                    df = pd.concat([df, to_add], ignore_index=True)

        request = youtube.commentThreads().list_next(request, response)

    return df


# try_df = get_comments("https://www.youtube.com/watch?v=Q-9tDyf3nv4", 100)
