from urllib.parse import urlparse, parse_qs
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter


class YouTubeProcessor:
    @staticmethod
    def get_video_id(url_or_id: str):
        if len(url_or_id) == 11 and "/" not in url_or_id:
            return url_or_id
        parsed_url = urlparse(url_or_id)
        if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
            return parse_qs(parsed_url.query).get("v", [None])[0]
        elif parsed_url.hostname == "youtu.be":
            return parsed_url.path[1:]
        return None

    @staticmethod
    def fetch_transcript(video_id: str) -> str:
        trs_api = YouTubeTranscriptApi()
        transcript_list = trs_api.fetch(video_id=video_id, languages=['en', 'hi'])
        return " ".join(
            chunk.text if hasattr(chunk, "text") else chunk["text"]
            for chunk in transcript_list
        )

    @staticmethod
    def split_text(transcript_text: str):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.create_documents([transcript_text])

    @staticmethod 
    def fetch_title(url_or_id: str) -> str:
        try:
            video_id = YouTubeProcessor.get_video_id(url_or_id)
            if not video_id:
                return f"Video {url_or_id[:11] if len(url_or_id) > 11 else url_or_id}"
            
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('title', f"Video {video_id}")
                
        except Exception as e:
            video_id = YouTubeProcessor.get_video_id(url_or_id)
            if video_id:
                return f"Video {video_id}"
            return f"Video {url_or_id[:11] if len(url_or_id) > 11 else url_or_id}"