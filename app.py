import streamlit as st
import requests
from datasets import load_dataset
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer

# Define Groq API endpoint and API key (replace 'YOUR_API_KEY' with your actual Groq API key)
GROQ_API_URL = "https://api.groq.com/llama/v1/chat/completions"  # Use correct Groq endpoint for your model
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]  # Access Groq API key from Streamlit secrets

# Load the dataset for context retrieval
ds = load_dataset("Amod/mental_health_counseling_conversations")
context_data = ds['train']  # Get the training data (or any other split as needed)

# Tokenizer for processing text input
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# Preprocess the data (you can improve this depending on your use case)
def preprocess_data(examples):
    # Assuming examples have 'Context' and 'Response' fields
    inputs = examples['Context']
    targets = examples['Response']
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=512)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Index the contexts for retrieval using FAISS (for efficient nearest neighbor search)
def index_contexts(context_data):
    # You can adjust embedding and dimension size based on your needs
    context_embeddings = [tokenizer.encode(context, return_tensors="pt", padding="max_length", truncation=True, max_length=512) for context in context_data['Context']]
    context_embeddings = torch.cat(context_embeddings, dim=0).numpy()  # Convert to NumPy array
    
    # Use FAISS for efficient indexing
    index = faiss.IndexFlatL2(context_embeddings.shape[1])  # Create FAISS index with correct dimensionality
    index.add(context_embeddings)  # Add embeddings to FAISS index
    return index, context_embeddings

# Retrieve the most relevant context for a query using FAISS
def retrieve_context(query, index, context_embeddings, top_k=5):
    # Ensure that the query is encoded with the same dimensions as the context embeddings
    query_embedding = tokenizer.encode(query, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    query_embedding = query_embedding.numpy()  # Convert query_embedding to NumPy array for FAISS compatibility
    
    # Perform the search using FAISS
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve the context data for the most similar results
    retrieved_contexts = [context_embeddings[i] for i in indices[0]]
    return retrieved_contexts

# Query the Groq API for Llama model inference
def query_groq_api(input_text, context):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "context": context,
        "query": input_text,
        "model": "llama"  # Specify the model you're using
    }
    response = requests.post(GROQ_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()['response']
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

# Streamlit interface
st.title("Mental Health Counseling Assistant with Groq and Llama")
st.write("Type your question, and get a response from the Llama model.")

# User input for interaction
user_input = st.text_area("How are you feeling today?", placeholder="Type here...")

# Index contexts for retrieval
index, context_embeddings = index_contexts(context_data)

# Retrieve relevant context and generate response from Llama
if user_input.strip():
    with st.spinner("Retrieving context and generating response..."):
        # Step 1: Retrieve the relevant context for the input query
        retrieved_context = retrieve_context(user_input, index, context_embeddings)
        retrieved_context_text = " ".join([str(c) for c in retrieved_context])

        # Step 2: Query the Groq API for Llama model's response
        response = query_groq_api(user_input, retrieved_context_text)
        if response:
            st.write("Model Response:", response)
else:
    st.info("Please enter your thoughts or feelings in the text area above.")
