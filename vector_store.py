import os
import pickle
import glob
import json
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class EmbeddingStore:
    def __init__(self, embedding_model_name="gemini-embedding-001"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)
        self.index_dir = "saved_indexes"
        self.metadata_file = os.path.join(self.index_dir, "video_metadata.json")
        os.makedirs(self.index_dir, exist_ok=True)

        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _get_index_path(self, video_id):
        return os.path.join(self.index_dir, f"{video_id}.faiss")

    def _get_meta_path(self, video_id):
        return os.path.join(self.index_dir, f"{video_id}_meta.pkl")

    def embedding_exists(self, video_id):
        return os.path.exists(self._get_index_path(video_id))

    def save_index(self, vector_store: FAISS, video_id: str, title: str = None):
        faiss_path = self._get_index_path(video_id)
        meta_path = self._get_meta_path(video_id)
        vector_store.save_local(faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(vector_store.docstore, f)

        self.metadata[video_id] = {"title": title or f"Video {video_id}"}
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=4)

    def load_index(self, video_id: str):
        faiss_path = self._get_index_path(video_id)
        if os.path.exists(faiss_path):
            return FAISS.load_local(
                faiss_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        return None

    def create_or_load_index(self, video_id: str, chunks, title=None):
        if self.embedding_exists(video_id):
            return self.load_index(video_id), False
        index = FAISS.from_documents(chunks, self.embeddings)
        self.save_index(index, video_id, title)
        return index, True

    def get_latest_video_id(self):
        indexes = glob.glob(os.path.join(self.index_dir, "*.faiss"))
        if not indexes:
            return None
        latest = max(indexes, key=os.path.getmtime)
        return os.path.basename(latest).replace(".faiss", "")

    def get_all_videos(self):
        return [(meta["title"], vid) for vid, meta in self.metadata.items()]
