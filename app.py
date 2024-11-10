import streamlit as st
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import requests
import numpy as np

# Retrieve Hugging Face token from Streamlit secrets
HUGGINGFACE_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN")

# Load the dataset from Hugging Face
ds = load_dataset("Amod/mental_health_counseling_conversations")

# Initialize model and tokenizer for embeddings (you can replace with the Llama model if needed)
tokenizer = AutoTokenizer.from_pretrained("facebook/llama-13b")  # Replace with appropriate Llama model
model = AutoModel.from_pretrained("facebook/llama-13b")  # Replace with appropriate Llama model

# Function to compute embeddings
def get_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.numpy()

# Preprocess dataset context
contexts = [example["Context"] for example in ds["train"]]
context_embeddings = get_embeddings(contexts, tokenizer, model)

# Set up FAISS index
dimension = context_embeddings.shape[1]  # Embedding dimension
index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
index.add(np.array(context_embeddings))  # Add embeddings to FAISS index

# Function to query Groq API with Llama
def query_groq_api(input_text):
    url = "https://api.groq.com/v1/query"  # Replace with actual Groq API endpoint
    headers = {
        "Authorization": f"Bearer {YOUR_GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-model-id",  # Replace with specific Llama model ID
        "input": input_text,
        "max_length": 256  # Adjust response length as needed
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()['generated_text']
    else:
        return f"Error: {response.status_code}"

# Function to generate RAG response
def generate_rag_response(user_query, index, tokenizer, model):
    # Step 1: Retrieve the most relevant context using FAISS
    query_embedding = get_embeddings([user_query], tokenizer, model)
    D, I = index.search(query_embedding, k=1)  # Find top-1 relevant context
    
    relevant_context = ds["train"][I[0][0]]["Context"]
    combined_input = relevant_context + "\nUser: " + user_query + "\nAssistant:"
    
    # Step 2: Query Llama model using Groq API
    rag_response = query_groq_api(combined_input)
    return rag_response

# Streamlit app interface
st.title("Mental Health Counseling Chatbot with RAG")
st.write("Type your thoughts or feelings, and get advice from the assistant.")

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# User input
user_input = st.text_area("How are you feeling today?", placeholder="Type here...")

# Handle user input and model response
if user_input.strip():
    # Add user input to conversation history
    st.session_state.conversation_history.append(f"User: {user_input}")
    
    with st.spinner("Generating response..."):
        response = generate_rag_response(user_input, index, tokenizer, model)
    
    # Add model response to conversation history
    st.session_state.conversation_history.append(f"Assistant: {response}")
    
    # Display the entire conversation history
    for message in st.session_state.conversation_history:
        st.write(message)

else:
    st.info("Please enter your thoughts or feelings in the text area above.")
