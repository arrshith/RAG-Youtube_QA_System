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

#STEP 1 : INDEXING

trs_api = YouTubeTranscriptApi()
transcript_list = trs_api.fetch(video_id="Gfr50f6ZBvo",languages=['en'])
transcript_text = " ".join(chunk.text if hasattr(chunk, 'text') else chunk['text'] for chunk in transcript_list)
#print(transcript_text)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript_text])
#print(len(chunks))

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
#print(vector_store)


#STEP 2 : RETRIEVING

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#print(retriever)


#STEP 3 : AUGMENTATION

"""llm = HuggingFaceEndpoint(
  repo_id="google/flan-t5-base",
  task="text2text-generation",
  max_new_tokens=256,
  temperature=0.1,
)"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0
)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

'''parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser'''

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


#STEP 4 : TEXT GENERATION

question = "Can you summarize the video"
response = rag_chain.invoke({'question': question})
print(response)