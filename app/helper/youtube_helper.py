from dotenv import load_dotenv
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from fastapi import HTTPException
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(f'{BASE_DIR}/../.env')
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("YOUTUBE_API_KEY environment variable is not set")

youtube = build("youtube", "v3", developerKey=API_KEY)


def fetch_youtube(query):
    try:
        search_response = youtube.search().list(
            q=query,
            type="video",
            part="id,snippet",
            maxResults=10
        ).execute()

        videos = []
        for search_result in search_response.get("items", []):
            video = {
                "title": search_result["snippet"]["title"],
                "description": search_result["snippet"]["description"],
                "thumbnail": search_result["snippet"]["thumbnails"]["default"]["url"],
                "video_id": search_result["id"]["videoId"]
            }
            videos.append(video)

        return {"videos": videos}
    except HttpError as e:
        raise HTTPException(status_code=e.resp.status, detail=str(e))

