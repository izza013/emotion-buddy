import streamlit as st
import requests
import os
from datasets import load_dataset
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer

# Get the Groq API key from environment variables or Streamlit secrets
api_key = os.environ.get("GROQ_API_KEY")  # or st.secrets["GROQ_API_KEY"] if using secrets
url = "https://api.groq.com/openai/v1/models"

# Define the headers for the API request
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Make the API request to get available models from Groq
response = requests.get(url, headers=headers)

# Check if the response is successful (status code 200)
if response.status_code == 200:
    models_data = response.json()
else:
    st.error(f"Error retrieving models: {response.status_code}")
    models_data = None

# Display the list of models if data is successfully retrieved
if models_data:
    st.write("Available Models from Groq API:")
    st.json(models_data)  # Display the raw model data as JSON

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
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Payload structure might need adjustment according to the actual API documentation
    payload = {
        "inputs": {
            "context": context,
            "query": input_text
        }
    }
    
    try:
        # Update the endpoint to the correct one if necessary
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json().get('response', "No response generated.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
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
