import os
import argparse
import pandas as pd
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY_ENV = os.getenv("YOUTUBE_API_KEY_2")

def get_youtube_service(api_key):
    """Create and return a YouTube API service object."""
    return build('youtube', 'v3', developerKey=api_key)

def search_videos(youtube, query, max_results=50, published_after=None, published_before=None, 
                  order='relevance', video_category=None, channel_id=None):
    """Search for videos using pagination."""
    all_videos = []
    next_page_token = None

    while len(all_videos) < max_results:
        remaining = max_results - len(all_videos)
        search_params = {
            'q': query,
            'part': 'snippet',
            'maxResults': min(50, remaining),  # Max 50 per request
            'type': 'video',
            'order': order,
            'pageToken': next_page_token
        }

        if published_after:
            search_params['publishedAfter'] = published_after
        if published_before:
            search_params['publishedBefore'] = published_before
        if channel_id:
            search_params['channelId'] = channel_id

        response = youtube.search().list(**search_params).execute()

        video_ids = [item['id']['videoId'] for item in response.get('items', [])]
        if not video_ids:
            break

        # Get video details
        videos_response = youtube.videos().list(
            part='snippet,contentDetails,statistics',
            id=','.join(video_ids)
        ).execute()

        videos = videos_response.get('items', [])
        if video_category:
            videos = [v for v in videos if v['snippet'].get('categoryId') == video_category]

        all_videos.extend(videos)
        next_page_token = response.get('nextPageToken')

        if not next_page_token:
            break

        # Optional: Delay to avoid quota issues
        time.sleep(0.5)

    return all_videos[:max_results]

def extract_video_data(videos):
    """Extract relevant data from video items."""
    video_data = []

    for video in videos:
        snippet = video['snippet']
        statistics = video.get('statistics', {})

        video_info = {
            'Title': snippet['title'],
            'Channel': snippet['channelTitle'],
            'Published Date': snippet['publishedAt'],
            'Video ID': video['id'],
            'URL': f"https://www.youtube.com/watch?v={video['id']}",
            'Description': snippet.get('description', ''),
            'View Count': statistics.get('viewCount', 'N/A'),
            'Like Count': statistics.get('likeCount', 'N/A'),
            'Comment Count': statistics.get('commentCount', 'N/A'),
            'Duration': video.get('contentDetails', {}).get('duration', 'N/A')
        }

        video_data.append(video_info)

    return video_data

def save_to_excel(video_data, output_file='youtube_videos.xlsx'):
    """Append video data to an Excel file instead of overwriting."""
    if not video_data:
        print("No videos found matching the criteria.")
        return False

    new_df = pd.DataFrame(video_data)

    if os.path.exists(output_file):
        try:
            existing_df = pd.read_excel(output_file)
            # Combine and remove duplicates based on 'Video ID'
            combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset='Video ID', keep='first')
        except Exception as e:
            print(f"⚠️ Error reading existing Excel file: {e}")
            combined_df = new_df
    else:
        combined_df = new_df

    try:
        combined_df.to_excel(output_file, index=False)
        print(f"✅ Appended data saved to {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving to Excel: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='YouTube Video Collector')
    parser.add_argument('--query', required=True, help='Search term')
    parser.add_argument('--max-results', type=int, default=50, help='Number of results to fetch')
    parser.add_argument('--published-after', help='Filter: Published after (YYYY-MM-DDThh:mm:ssZ)')
    parser.add_argument('--published-before', help='Filter: Published before (YYYY-MM-DDThh:mm:ssZ)')
    parser.add_argument('--order', default='relevance', choices=['date', 'rating', 'relevance', 'title', 'videoCount', 'viewCount'])
    parser.add_argument('--category', help='YouTube category ID (optional)')
    parser.add_argument('--channel-id', help='Filter by Channel ID (optional)')
    parser.add_argument('--output', default='youtube_videos.xlsx', help='Output Excel filename')
    parser.add_argument('--api-key', default=API_KEY_ENV, help='YouTube API key (optional if in .env)')
    args = parser.parse_args()

    if not args.api_key:
        print("❌ API key not provided. Set YOUTUBE_API_KEY in .env or pass via --api-key")
        return

    try:
        youtube = get_youtube_service(args.api_key)
        videos = search_videos(
            youtube,
            query=args.query,
            max_results=args.max_results,
            published_after=args.published_after,
            published_before=args.published_before,
            order=args.order,
            video_category=args.category,
            channel_id=args.channel_id
        )

        video_data = extract_video_data(videos)
        save_to_excel(video_data, args.output)

    except HttpError as e:
        print(f"HTTP error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()
