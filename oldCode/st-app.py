import streamlit as st
from urllib.parse import urlparse, parse_qs
#from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Helper: Extract video ID

def get_video_id(url_or_id):
    if len(url_or_id) == 11 and "/" not in url_or_id:
        return url_or_id
    parsed_url = urlparse(url_or_id)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        return parse_qs(parsed_url.query).get('v', [None])[0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    else:
        return None

# Streamlit UI

st.title("RAG Based YouTube Video Q&A System")
st.markdown("Enter a YouTube video URL or ID and ask questions or doubts about it.")

video_input = st.text_input("YouTube URL or Video ID")
user_question = st.text_input("Your Question")

if st.button("Get Answer"):

    if not video_input or not user_question:
        st.warning("Please enter both the YouTube URL/ID and your question.")
    else:
        video_id = get_video_id(video_input)
        if not video_id:
            st.error("Invalid YouTube URL or Video ID.")
        else:
            try:
                # STEP 1 : INDEXING
                trs_api = YouTubeTranscriptApi()
                transcript_list = trs_api.fetch(video_id=video_id, languages=['en'])
                transcript_text = " ".join(
                    chunk.text if hasattr(chunk, 'text') else chunk['text'] 
                    for chunk in transcript_list
                )

                splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript_text])

                embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
                vector_store = FAISS.from_documents(chunks, embeddings)

                # STEP 2 : RETRIEVING
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

                # STEP 3 : AUGMENTATION
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    temperature=0,
                    GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
                )

                prompt = PromptTemplate(
                    template="""
                      You are a helpful assistant.
                      Answer ONLY from the provided transcript context.
                      If the context is insufficient, just say you don't know.

                      {context}
                      Question: {question}
                    """,
                    input_variables=['context', 'question']
                )

                def format_docs(retrieved_docs):
                    return "\n\n".join(doc.page_content for doc in retrieved_docs)

                def get_context(inputs):
                    question = inputs['question']
                    retrieved_docs = retriever.invoke(question)
                    return format_docs(retrieved_docs)

                rag_chain = (
                    {
                        "context": RunnableLambda(get_context),
                        "question": RunnablePassthrough()
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                # STEP 4 : TEXT GENERATION
                response = rag_chain.invoke({'question': user_question})
                st.markdown("**Answer:**")
                st.write(response)

            except TranscriptsDisabled:
                st.error("Transcripts are disabled for this video.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
