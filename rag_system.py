from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from video_processor import YouTubeProcessor
from vector_store import EmbeddingStore

load_dotenv()

class YouTubeRAGApp:
    def __init__(self):
        self.processor = YouTubeProcessor()
        self.embedding_store = EmbeddingStore()
        self.retriever = None

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
            streaming=True 
        )

        self.prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            The transcript context may be in Hindi.
            Answer the question in ENGLISH ONLY based on the context.
            If the context does not have enough information, say "I don't know."

            Context:
            {context}

            Question: {question}
            """,
            input_variables=["context", "question"]
        )

    def load_embeddings(self, video_id=None):
        if video_id:
            if not self.embedding_store.embedding_exists(video_id):
                raise ValueError("No saved embeddings found for this video.")
            vector_store = self.embedding_store.load_index(video_id)
            self.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            return video_id, False
        else:
            latest_id = self.embedding_store.get_latest_video_id()
            if not latest_id:
                raise ValueError("No previous embeddings found.")
            vector_store = self.embedding_store.load_index(latest_id)
            self.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            return latest_id, True

    def create_embeddings(self, video_id, title=None):
        transcript_text = self.processor.fetch_transcript(video_id)
        chunks = self.processor.split_text(transcript_text)
        vector_store, created = self.embedding_store.create_or_load_index(video_id, chunks, title)
        self.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        return created

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_rag_chain(self):
        def get_context(inputs):
            question = inputs["question"]
            retrieved_docs = self.retriever.invoke(question)
            return self._format_docs(retrieved_docs)

        return (
            {
                "context": RunnableLambda(get_context),
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def answer_question(self, question):
        if not self.retriever:
            raise ValueError("Retriever not loaded. Please load or create embeddings first.")
        rag_chain = self._build_rag_chain()
        return rag_chain.invoke({"question": question})
