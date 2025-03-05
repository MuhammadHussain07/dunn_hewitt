import json
import numpy as np
from dotenv import load_dotenv
import os
import faiss
import openai
import streamlit as st
from langchain_openai import OpenAIEmbeddings


# 🔹 Load .env file and OpenAI API key
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]

# 🔴 Raise error if API key is missing
if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing! Check your .env file.")

# Initialize OpenAI client
client = openai.OpenAI(api_key=openai_api_key)

# Load JSON Data Correctly
def load_json(file_path="final_persons.json"):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["persons"]

# Generate and Store FAISS Index
def create_vector_store(json_data):
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)  # ✅ FIXED
    texts = [entry["full_name"][:200] for entry in json_data]
    vectors = np.array([embeddings_model.embed_query(text) for text in texts], dtype=np.float32)

    # Create FAISS index
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # Save FAISS index and metadata
    faiss.write_index(index, "faiss_index.bin")
    with open("metadata.json", "w", encoding="utf-8") as file:
        json.dump({"texts": texts, "json_data": json_data}, file, indent=4)

    return index, texts, json_data

# Load FAISS Index and Metadata
def load_vector_store():
    index = faiss.read_index("faiss_index.bin")
    with open("metadata.json", "r", encoding="utf-8") as file:
        metadata = json.load(file)
    return index, metadata["texts"], metadata["json_data"]

# Find Best Match in JSON Data
def find_best_match(query, index, texts, json_data):
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)  # ✅ FIXED
    query_vector = np.array([embeddings_model.embed_query(query[:200])], dtype=np.float32)

    _, I = index.search(query_vector, k=1)
    match_index = I[0][0]

    if match_index == -1:
        return "Sorry, I don't have information on that."

    return json_data[match_index]

# Generate AI-Powered Response with OpenAI (GPT-4o)
def get_chat_response(user_query):
    index, texts, json_data = load_vector_store()
    best_match = find_best_match(user_query, index, texts, json_data)

    best_match_text = json.dumps(best_match, indent=4) if isinstance(best_match, dict) else best_match

    # ✅ FIXED: Pass API key explicitly
    response = client.chat.completions.create( 
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a genealogy assistant providing accurate historical data."},
            {"role": "user", "content": f"User asked: {user_query}. Best match found: {best_match_text}"}
        ]
    )

    return response.choices[0].message.content

# Streamlit UI
st.title("📜 Historical Records Chatbot (GPT-4o)")

user_query = st.text_input("Ask about a person in history:")
if user_query:
    response = get_chat_response(user_query)
    st.write("📖 Chatbot:", response)

# Run this once to generate FAISS Index (Only needed for the first time)
if __name__ == "__main__":
    data = load_json()
    create_vector_store(data)
    print("✅ FAISS index and metadata saved!")
