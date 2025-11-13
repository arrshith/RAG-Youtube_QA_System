# RAG YouTube Q&A Chatbot

A **Retrieval-Augmented Generation (RAG)** application that allows you to ask questions about any YouTube video — even if it has long content. The system automatically:

* extracts the transcript (English or Hindi),
* splits it into chunks,
* embeds and stores the chunks using **FAISS**,
* retrieves relevant segments for a query, and
* answers your question using **Gemini 2.5 Pro**.

This app comes with a **Streamlit UI**, supports **saved video embeddings**, and allows fast reuse of previously processed videos.

---

## 🚀 Features

### ✅ **Automated transcript extraction**

* Supports both English and Hindi using `youtube_transcript_api`.

### ✅ **Smart text splitting**

* Uses `RecursiveCharacterTextSplitter` to create RAG-friendly chunks.

### ✅ **Reusable embeddings**

* Embeddings stored locally using **FAISS**.
* Automatically loads past processed videos.

### ✅ **Supports YouTube URLs & Video IDs**

* Automatically extracts the correct video ID.

### ✅ **Title fetching using yt-dlp**

* Works even when transcripts are long or messy.

### ✅ **LLM-powered answering**

* Uses Google Gemini for structured retrieval-based responses.

### ✅ **Streamlit UI**

* Simple and clean interface for entering URLs, selecting stored videos, and asking questions.

---

## 📁 Project Structure

```
./oldCode/              # Legacy code (not used in current version)
./rag_system.py         # Main RAG pipeline (LLM, retriever, embeddings)
./video_processor.py    # Transcript extraction + preprocessing
./vector_store.py       # Embedding storage & FAISS index handling
./main.py               # Streamlit UI
./requirements.txt      # Dependencies
./saved_indexes/        # Auto-created embedding files
```

---

## 🧠 System Architecture

### **1. Transcript Extraction** (`video_processor.py`)

* Extracts video ID (supports normal & short URLs)
* Downloads transcript in EN/HI
* Splits transcript into overlapping chunks

### **2. Embedding Storage** (`vector_store.py`)

* Embeds chunks using `GoogleGenerativeAIEmbeddings`
* Saves FAISS index + metadata
* Automatically loads or creates new embeddings

### **3. RAG Pipeline** (`rag_system.py`)

* Creates a retriever with top-K similarity search
* Builds a RAG chain with:

  * context injection
  * custom prompt
  * Gemini 2.5 Pro (temperature = 0)

### **4. Streamlit Frontend** (`main.py`)

* Select previously saved videos
* Enter new YouTube URLs or IDs
* Ask questions and view answers

---

## 🛠️ Installation & Setup (Local Machine)

### **1. Clone the repository**

```bash
git clone https://github.com/arrshith/RAG-Youtube_QA_System.git
cd RAG-Youtube_QA_System
```

---

### **2. Create a virtual environment (Recommended)**

```bash
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
```

---

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

> ⚠️ FAISS may require additional system packages depending on your OS.

---

### **4. Set up your API Key**

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_api_key_here
```

You can get the key from:
**[https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)**

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run main.py
```

This will open a local UI in your browser:

```
http://localhost:8501
```

---

## 🧪 How to Use the Application

### **Step 1 — Choose an option**

You have two options:

1. **Select a previously processed video** from the dropdown
2. **Enter a new YouTube URL or direct video ID**

### **Step 2 — Ask any question**

Examples:

* *"What is the summary of this video?"*
* *"What did the speaker say about machine learning pipelines?"*
* *"Explain the steps mentioned between 5:00 and 10:00."*

### **Step 3 — Get your answer**

* If it's a new video: the system creates embeddings
* If already processed: loads embeddings instantly
* Uses RAG to generate accurate answers from the transcript

---

## 📦 Where Are Embeddings Stored?

Inside:

```
./saved_indexes/
```

Files generated:

* `VIDEO_ID.faiss` → FAISS vector index
* `VIDEO_ID_meta.pkl` → metadata store
* `video_metadata.json` → all video titles

You can delete this folder anytime to reset.

---

## 🔥 Highlights

### 💨 Fast Retrieval

FAISS makes question-answering fast even for **very long videos**.

### 🎯 Accurate Responses

LLM answers strictly using transcript context.

### 📁 Persistent Video Memory

Re-run the app and all previously processed videos are still available.

### 🧩 Modular Code

Video extraction, embedding, RAG logic, and UI are cleanly separated.

---

## 🧹 Optional Cleanup

If you want to restart fresh, delete:

```
saved_indexes/
```

And optionally remove `__pycache__` folders.

---

## 🤝 Contributing

Feel free to open issues or submit PRs to improve:

* RAG pipeline
* UI design
* Support for multi-language output
* Support for subtitles without transcripts

---

## 📜 License

This project is open-source under the MIT License.

---

## ⭐️ Support

If you like this project, consider starring the repo!

Happy coding! 🚀

(README created using AI)
