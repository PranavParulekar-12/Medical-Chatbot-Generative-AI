from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *

import google.generativeai as genai

app = Flask(__name__)
load_dotenv()

# Load API Keys
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Embeddings & Pinecone index
embeddings = download_hugging_face_embeddings()
index_name = "chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create Gemini wrapper (instead of OpenAI)
class GeminiLLM:
    def __init__(self, temperature=0.4, max_tokens=500):
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt_text):
        response = self.model.generate_content(prompt_text)
        return response.text

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}\n\nContext:\n{context}"),
    ]
)

# LLM and Chain Setup
llm = GeminiLLM()

def gemini_chain(input_text):
    # Handle greetings
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
    if input_text.lower().strip() in greetings:
        return "Hello! How can I assist you today? 👋"

    # Retrieve documents from Pinecone
    docs = retriever.invoke(input_text)

    if not docs or all(len(doc.page_content.strip()) == 0 for doc in docs):
        print("❌ No relevant documents found. Asking Gemini directly.")
        answer = llm.invoke(input_text)
        return f"🧠 AI Answer:\n{answer}"


    # Build final prompt from documents
    combined_doc_content = "\n\n".join([doc.page_content for doc in docs])
    final_prompt = prompt.format(input=input_text, context=combined_doc_content)
    final_prompt_str = str(final_prompt)

    # Generate Gemini response using PDF context
    answer = llm.invoke(final_prompt_str)
    return f"\n{answer}"




# Routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)
    response = gemini_chain(msg)
    print("Response:", response)
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
