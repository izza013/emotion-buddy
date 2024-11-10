import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Retrieve Hugging Face token from Streamlit secrets
HUGGINGFACE_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN")

# Check if the token is available and valid
if not HUGGINGFACE_TOKEN:
    st.error("Hugging Face token not found. Please set the HUGGINGFACE_TOKEN in Streamlit Secrets.")
    st.stop()

# Function to load model and tokenizer
def load_model(model_path):
    try:
        # Load tokenizer and model using Hugging Face token for gated models
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=HUGGINGFACE_TOKEN)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, use_auth_token=HUGGINGFACE_TOKEN)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path from Hugging Face
model_path = "Izza-shahzad-13/fine-tuned-flan-t5"  # Replace with actual model path

# Load tokenizer and model
tokenizer, model = load_model(model_path)
if model and tokenizer:
    model.to(device)
else:
    st.stop()  # Stop the app if model or tokenizer failed to load

# Function to generate response
def generate_response(input_text, conversation_history):
    # Concatenate conversation history with current input
    input_text = "\n".join(conversation_history) + "\nUser: " + input_text + "\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        output = model.generate(
            inputs['input_ids'],
            max_length=1024,
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
st.write("Type your thoughts or feelings, and let the model respond. The conversation will continue as you interact with the model.")

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
        response = generate_response(user_input, st.session_state.conversation_history)
    
    # Add model response to conversation history
    st.session_state.conversation_history.append(f"Assistant: {response}")
    
    # Display the entire conversation history
    for message in st.session_state.conversation_history:
        st.write(message)

else:
    st.info("Please enter your thoughts or feelings in the text area above.")
