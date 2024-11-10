import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# Retrieve the Hugging Face token from environment variable
HUGGINGFACE_TOKEN = os.getenv("HF_ACCESS_TOKEN")

# Check if the token is available
if not HUGGINGFACE_TOKEN:
    st.error("Hugging Face token not found. Please set the HUGGINGFACE_TOKEN environment variable.")
    st.stop()  # Stop the app if the token is not found

# Function to load model and tokenizer (local or Hugging Face with token)
def load_model(model_path):
    try:
        # Load tokenizer and model with authentication token if required
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=HUGGINGFACE_TOKEN)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, use_auth_token=HUGGINGFACE_TOKEN)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to your model (either a local path or a Hugging Face model name)
model_path = "Izza-shahzad-13/fine-tuned-flan-t5"  # Use your Hugging Face model identifier

# Load tokenizer and model
tokenizer, model = load_model(model_path)
if model and tokenizer:
    model.to(device)
else:
    st.stop()  # Stop the app if model or tokenizer failed to load

# Function to generate response from the model
def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model.generate(
            inputs['input_ids'],
            max_length=500,
            num_beams=4,
            top_p=0.9,
            top_k=50,
            temperature=0.7,
            do_sample=True,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Streamlit app interface
st.title("FLAN-T5 Mental Health Counseling Assistant")
st.write("Type your thoughts or feelings, and let the model respond.")

# User input for interaction
user_input = st.text_area("How are you feeling today?", placeholder="Type here...")

# Generate and display model response when input is provided
if user_input.strip():  # Check if input is not empty
    with st.spinner("Generating response..."):
        response = generate_response(user_input)
    st.write("Model Response:", response)
else:
    st.info("Please enter your thoughts or feelings in the text area above.")  
