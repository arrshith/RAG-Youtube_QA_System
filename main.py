import streamlit as st
from rag_system import YouTubeRAGApp
from video_processor import YouTubeProcessor
from youtube_transcript_api import TranscriptsDisabled

st.title("RAG-based YouTube Q&A System")
st.markdown(
    "Select a previously saved video or enter a new YouTube URL below."
)

app = YouTubeRAGApp()

saved_videos = app.embedding_store.get_all_videos()
saved_titles = [title for title, _ in saved_videos]

selected_video = None
if saved_titles:
    selected_title = st.selectbox("Choose from saved videos:", ["None"] + saved_titles)
    if selected_title != "None":
        selected_video = next(vid for title, vid in saved_videos if title == selected_title)

video_input = st.text_input("Enter a new YouTube URL or Video ID")
user_question = st.text_input("Your Question")

if st.button("Get Answer"):
    try:
        if selected_video:
            video_id = selected_video
            app.load_embeddings(video_id)
            st.info(f"Loaded embeddings for the video: {selected_title}")

        elif video_input.strip():
            video_id = YouTubeProcessor.get_video_id(video_input)
            video_title = YouTubeProcessor.fetch_title(video_input)
            created = app.create_embeddings(video_id, title=video_title)
            if created:
                st.success(f"New embeddings created for the video: {video_title}")
            else:
                st.info(f"Existing embeddings loaded for the video: {video_title}")

        else:
            video_id, _ = app.load_embeddings()
            st.info(f"Auto-loaded last used video: {video_id}")

        answer = app.answer_question(user_question)
        st.markdown("Answer:")
        st.write(answer)

    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except Exception as e:
        st.error(f"Error: {e}")
