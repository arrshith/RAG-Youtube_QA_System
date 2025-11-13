import streamlit as st
from rag_system import YouTubeRAGApp
from video_processor import YouTubeProcessor
from youtube_transcript_api import TranscriptsDisabled

st.title("RAG-based YouTube Q&A System")
st.markdown("Select a previously saved video or enter a new YouTube URL below.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None

if "current_video_title" not in st.session_state:
    st.session_state.current_video_title = None

# Initialize app
app = YouTubeRAGApp()

# Sidebar for video selection
with st.sidebar:
    st.header("Video Selection")
    
    saved_videos = app.embedding_store.get_all_videos()
    saved_titles = [title for title, _ in saved_videos]

    selected_video = None
    if saved_titles:
        selected_title = st.selectbox("Choose from saved videos:", ["None"] + saved_titles)
        if selected_title != "None":
            selected_video = next(vid for title, vid in saved_videos if title == selected_title)

    video_input = st.text_input("Enter a new YouTube URL or Video ID")
    
    if st.button("Load Video", use_container_width=True):
        try:
            if selected_video:
                video_id = selected_video
                app.load_embeddings(video_id)
                st.session_state.current_video_id = video_id
                st.session_state.current_video_title = selected_title
                st.session_state.messages = []  # Clear chat history for new video
                st.session_state.app = app  # Store app in session state
                st.success(f"✓ Loaded: {selected_title}")

            elif video_input.strip():
                video_id = YouTubeProcessor.get_video_id(video_input)
                if not video_id:
                    st.error("Invalid YouTube URL or Video ID")
                else:
                    try:
                        video_title = YouTubeProcessor.fetch_title(video_input)
                    except Exception:
                        video_title = f"Video {video_id}"
                    
                    created = app.create_embeddings(video_id, title=video_title)
                    st.session_state.current_video_id = video_id
                    st.session_state.current_video_title = video_title
                    st.session_state.messages = []  # Clear chat history for new video
                    st.session_state.app = app  # Store app in session state
                    
                    if created:
                        st.success(f"✓ New embeddings created for {video_title}")
                    else:
                        st.success(f"✓ Loaded existing embeddings for {video_title}")
            else:
                st.warning("Please select or enter a video")
                
        except TranscriptsDisabled:
            st.error("❌ Transcripts are disabled for this video")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    # Display current video info
    if st.session_state.current_video_title:
        st.divider()
        st.info(f"**Current Video:**\n{st.session_state.current_video_title}")
        
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# Main chat interface
if st.session_state.current_video_id:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_question := st.chat_input("Ask a question about the video..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Get answer from RAG system
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = app.answer_question(user_question)
                st.markdown(answer)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            with st.chat_message("assistant"):
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

else:
    st.info("👈 Please load a video from the sidebar to start chatting")