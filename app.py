import streamlit as st
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the dataset for context retrieval
ds = load_dataset("Amod/mental_health_counseling_conversations")
context_data = ds['train']  # Get the training data (or any other split as needed)

# Load the Llama tokenizer and model (meta-llama/Llama-3.2-1B)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

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
import faiss
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

# Local inference using Llama-3.2-1B
def generate_response(input_text, context):
    # Prepare the input for the model
    input_with_context = f"Context: {context} \nQuery: {input_text}"

    # Tokenize the input
    inputs = tokenizer(input_with_context, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generate the output from the model
    with torch.no_grad():
        output = model.generate(**inputs, max_length=512, num_beams=5, no_repeat_ngram_size=2)

    # Decode the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Streamlit interface
st.title("Mental Health Counseling Assistant with Llama-3.2-1B")
st.write("Type your question, and get a response from the Llama-3.2-1B model.")

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

        # Step 2: Generate the response using Llama-3.2-1B
        response = generate_response(user_input, retrieved_context_text)
        st.write("Model Response:", response)
else:
    st.info("Please enter your thoughts or feelings in the text area above.")
