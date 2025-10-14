import streamlit as st
import os
import json
from datetime import datetime
import base64

try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch
except ImportError:
    class T5Tokenizer:
        @staticmethod
        def from_pretrained(name): return None
    class T5ForConditionalGeneration:
        @staticmethod
        def from_pretrained(name, **kwargs): return None
    class torch:
        @staticmethod
        def device(device_type): return None
        @staticmethod
        def cuda(): return None
        @staticmethod
        def no_grad(): return None
    st.warning("LLM Dependencies (transformers, torch) not found. Running in UI-only mode.")
    st.warning("Please install them: pip install transformers torch")

class Config:
    MODEL_NAME = "google/flan-t5-base"
    MAX_INPUT_LENGTH = 512
    MAX_OUTPUT_LENGTH = 256
    TEMPERATURE = 0.7
    NUM_BEAMS = 4

@st.cache_resource
def load_model():
    """Load FLAN-T5 model and tokenizer with optimization"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try loading from Hugging Face Hub first (for deployment)
        hf_model_id = "Afsa20/Farmsmart_Growmate"
        try:
            tokenizer = T5Tokenizer.from_pretrained(hf_model_id)
            model = T5ForConditionalGeneration.from_pretrained(
                hf_model_id,
                torch_dtype=torch.float16,  # Use float16 for smaller size
                low_cpu_mem_usage=True,
                device_map="auto" if device.type == 'cuda' else None
            )
            
            # Try to load model info if available
            model_info = {}
            try:
                from huggingface_hub import hf_hub_download
                model_info_path = hf_hub_download(repo_id=hf_model_id, filename="model_info.json")
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
            except:
                pass
            
            return tokenizer, model, "Fine-tuned FLAN-T5 (from HF Hub)", device, model_info
        
        except Exception as hf_error:
            print(f"Hugging Face model loading failed: {hf_error}")
            
            # Check for local fine-tuned model as fallback
        trained_model_dir = "trained_model"
        if os.path.exists(trained_model_dir):
            config_path = os.path.join(trained_model_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                model_info_path = os.path.join(trained_model_dir, "model_info.json")
                model_info = {}
                if os.path.exists(model_info_path):
                    with open(model_info_path, 'r') as f:
                        model_info = json.load(f)
                
                tokenizer = T5Tokenizer.from_pretrained(trained_model_dir)
                model = T5ForConditionalGeneration.from_pretrained(
                    trained_model_dir,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto" if device.type == 'cuda' else None
                )
                
                return tokenizer, model, "Fine-tuned FLAN-T5 (Local)", device, model_info
        
        # Fallback to base FLAN-T5 model
        tokenizer = T5Tokenizer.from_pretrained(Config.MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(
            Config.MODEL_NAME,
            torch_dtype=torch.float16,  # Use float16 for smaller size
            low_cpu_mem_usage=True,
            device_map="auto" if device.type == 'cuda' else None
        )
        
        return tokenizer, model, "Base FLAN-T5", device, {}
        
    except Exception as e:
        # st.error(f"Model loading failed: {e}")
        return None, None, None, None, {}


def generate_response(question, tokenizer, model, model_type, device):
    """Generate response using FLAN-T5"""
    if not (tokenizer and model):
        return "Model not fully loaded. Please check installation and configuration."

    try:
        # Use EXACT format from training
        if "fine-tuned" in model_type.lower():
            input_text = f"Answer this hydroponic farming question: {question}"
        else:
            input_text = f"Answer this hydroponic farming question in detail: {question}"
        
        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=Config.MAX_INPUT_LENGTH,
            truncation=True,
            padding=True
        )
        
        # Move to device if using CUDA
        if device.type == 'cuda':
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model.to(device)
        
        # Generate response
        with torch.no_grad():
            if "fine-tuned" in model_type.lower():
                outputs = model.generate(
                    **inputs,
                    max_length=Config.MAX_OUTPUT_LENGTH,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_length=Config.MAX_OUTPUT_LENGTH,
                    num_beams=Config.NUM_BEAMS,
                    early_stopping=True,
                    do_sample=True,
                    temperature=Config.TEMPERATURE,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Fallback response if empty
        if not response or len(response) < 10:
            response = "I understand you're asking about hydroponic farming. While I don't have a specific answer for that question, I recommend checking pH levels (5.5-6.5), ensuring proper nutrients, and maintaining good water circulation in your hydroponic system."
        
        return response
        
    except Exception as e:
        return f"I apologize, but I'm having trouble processing your question right now. Error: {str(e)}. Please try rephrasing your question about hydroponic farming."


def ensure_model_loaded():
    """Auto-load model and store in session state."""
    if "model_loaded" not in st.session_state or not st.session_state.model_loaded:
        with st.spinner("Loading FarmSmart LLM..."):
            tokenizer, model, model_type, device, model_info = load_model()
            if tokenizer and model:
                st.session_state.tokenizer = tokenizer
                st.session_state.model = model
                st.session_state.model_type = model_type
                st.session_state.device = device
                st.session_state.model_info = model_info
                st.session_state.model_loaded = True
            else:
                # If model load failed (e.g., dependencies missing), set to True to prevent re-looping
                st.session_state.model_loaded = True 
                st.session_state.tokenizer = None
                st.session_state.model = None
                st.error("LLM Core failed to initialize. Functionality is limited.")


# --- Streamlit UI and Logic ---

def main():
    # Load favicon
    favicon_path = os.path.join("Assets", "loader.png")
    if os.path.exists(favicon_path):
        from PIL import Image
        favicon_img = Image.open(favicon_path)
        st.set_page_config(
            page_title="FarmSmart - GrowMate",
            page_icon=favicon_img,
            layout="wide",
            initial_sidebar_state="collapsed"
        )
    else:
        st.set_page_config(
            page_title="FarmSmart - GrowMate",
            page_icon=":seedling:",
            layout="wide",
            initial_sidebar_state="collapsed"
        )

    # Enhanced modern CSS with animations and better styling
    st.markdown("""
    <style>
        /* Hide all Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        div[data-testid="stDecoration"] {visibility: hidden;}
        div[data-testid="stToolbar"] {visibility: hidden;}
        
        /* Remove ALL default Streamlit spacing and margins */
        html, body {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Smooth scrolling */
        html {
            scroll-behavior: smooth;
        }
        
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e8f5e9 100%);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        .main {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        .main .block-container {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100% !important;
        }
        
        div[data-testid="stAppViewContainer"] {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        div[data-testid="stAppViewContainer"] > div {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        /* Remove top spacing from all containers */
        .stApp > div {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Nuclear option - remove all possible spacing */
        * {
            box-sizing: border-box;
        }
        
        /* Target specific Streamlit containers */
        div[class*="main"], 
        div[class*="block-container"],
        div[class*="element-container"],
        section[data-testid="stSidebar"] {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Ensure header container has no spacing */
        .stApp > div:first-child {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Remove any Streamlit injected spacing */
        .stApp [data-testid="stAppViewContainer"] {
            padding-top: 0 !important;
        }
        
        .stApp [data-testid="stAppViewContainer"] > div {
            padding-top: 0 !important;
        }
        
        /* Ensure chat starts immediately after header */
        .header + div {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        /* Remove any spacing from chat message containers */
        [data-testid="stChatMessage"]:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        /* Remove spacing from the element that contains chat messages */
        div[data-testid="stChatMessage"]:first-child > div {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        /* Target the Streamlit chat message wrapper */
        .stChatMessage {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        .stChatMessage:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        /* Remove default spacing from chat message containers */
        div[data-testid="stVerticalBlock"] {
            padding-top: 0 !important;
            margin-top: 0 !important;
            gap: 0 !important;
        }
        
        div[data-testid="stVerticalBlock"]:first-child {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Target the container that holds all chat messages */
        section[data-testid="stSidebar"] ~ div {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Enhanced Header with animations */
        @keyframes slideDown {
            from {
                transform: translateY(-100%);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.5;
            }
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 24px;
            background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
            border-bottom: 2px solid #10b981;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin: 0;
            position: sticky;
            top: 0;
            z-index: 1000;
            animation: slideDown 0.5s ease-out;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 700;
            font-size: 18px;
            color: #1f2937;
            transition: transform 0.3s ease;
            cursor: pointer;
        }
        
        .logo:hover {
            transform: scale(1.05);
        }
        
        .logo-icon {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 16px;
            box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3);
            transition: all 0.3s ease;
        }
        
        .logo-icon:hover {
            transform: rotate(10deg);
            box-shadow: 0 6px 12px -1px rgba(16, 185, 129, 0.4);
        }
        
        .center-title {
            text-align: center;
            flex-grow: 1;
        }
        
        .main-title {
            font-size: 20px;
            font-weight: 700;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
        }
        
        .subtitle {
            font-size: 13px;
            color: #6b7280;
            margin: 4px 0 0 0;
            font-weight: 500;
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: #10b981;
            font-weight: 600;
            padding: 8px 16px;
            background-color: #ecfdf5;
            border-radius: 20px;
            transition: all 0.3s ease;
        }
        
        .status:hover {
            background-color: #d1fae5;
            transform: scale(1.05);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            animation: pulse 2s ease-in-out infinite;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.6);
        }
        
        /* Chat area - for additional messages flowing from welcome */
        .chat-area {
            max-width: 800px;
            margin: 0 auto;
            padding: 0px 20px 20px 20px;
            min-height: calc(100vh - 200px);
        }
        
        /* Ensure first user message flows directly after welcome */
        .chat-area [data-testid="stChatMessage"]:first-child {
            margin-top: 0px !important;
            padding-top: 0px !important;
        }
        
        /* Aggressive removal of gaps in chat area */
        .chat-area > div {
            margin-top: 0px !important;
            padding-top: 0px !important;
        }
        
        .chat-area > div > div {
            margin-top: 0px !important;
            padding-top: 0px !important;
        }
        
        /* Target Streamlit's vertical block containers in chat area */
        .chat-area div[data-testid="stVerticalBlock"] {
            margin-top: 0px !important;
            padding-top: 0px !important;
            gap: 0px !important;
        }
        
        /* Remove spacing from all elements within chat area */
        .chat-area * {
            margin-top: 0px !important;
        }
        
        /* Specific targeting for the first element in chat area */
        .chat-area > *:first-child {
            margin-top: 0px !important;
            padding-top: 0px !important;
        }
        
        /* Message bubble animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInLeft {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes fadeInRight {
            from {
                opacity: 0;
                transform: translateX(20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* Enhanced message bubbles */
        [data-testid="stChatMessage"] {
            margin-bottom: 20px;
            animation: fadeInUp 0.4s ease-out;
        }
        
        /* First message - start immediately */
        [data-testid="stChatMessage"]:first-child {
            margin-top: 0px;
        }
        
        /* Bot messages - light gray, left aligned with shadow */
        [data-testid="stChatMessage"][data-testid="assistant"] [data-testid="stMarkdownContainer"],
        [data-testid="stChatMessage"]:not([data-testid="user"]) [data-testid="stMarkdownContainer"] {
            background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%) !important;
            border-radius: 18px 18px 18px 4px !important;
            padding: 14px 18px !important;
            color: #374151 !important;
            margin: 0 !important;
            max-width: 80% !important;
            display: inline-block !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
            border: 1px solid #e5e7eb !important;
            animation: fadeInLeft 0.5s ease-out !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }
        
        [data-testid="stChatMessage"][data-testid="assistant"] [data-testid="stMarkdownContainer"]:hover,
        [data-testid="stChatMessage"]:not([data-testid="user"]) [data-testid="stMarkdownContainer"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
        }
        
        /* User messages - green gradient background, right aligned */
        [data-testid="stChatMessage"][data-testid="user"] [data-testid="stMarkdownContainer"] {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
            border-radius: 18px 18px 4px 18px !important;
            padding: 14px 18px !important;
            color: white !important;
            margin: 0 !important;
            max-width: 80% !important;
            display: inline-block !important;
            float: right !important;
            box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3), 0 2px 4px -1px rgba(16, 185, 129, 0.2) !important;
            animation: fadeInRight 0.5s ease-out !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }
        
        [data-testid="stChatMessage"][data-testid="user"] [data-testid="stMarkdownContainer"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.4), 0 4px 6px -2px rgba(16, 185, 129, 0.3) !important;
        }
        
        /* Message text */
        [data-testid="stChatMessage"] p {
            margin: 0 !important;
            line-height: 1.4 !important;
            font-size: 14px !important;
        }
        
        /* Timestamps */
        .timestamp {
            font-size: 11px;
            color: #9ca3af;
            margin-top: 4px;
        }
        
        /* Enhanced input area */
        .input-area {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(to top, rgba(255, 255, 255, 1) 0%, rgba(255, 255, 255, 0.98) 100%);
            border-top: 2px solid #10b981;
            padding: 20px 24px;
            z-index: 1000;
            box-shadow: 0 -4px 6px -1px rgba(0, 0, 0, 0.1), 0 -2px 4px -1px rgba(0, 0, 0, 0.06);
            backdrop-filter: blur(10px);
        }
        
        .input-container {
            max-width: 800px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        /* Chat input styling with glow effect */
        .stChatInput {
            flex: 1;
        }
        
        .stChatInput > div > div > textarea {
            border-radius: 24px !important;
            border: 2px solid #e5e7eb !important;
            background-color: #ffffff !important;
            padding: 12px 20px !important;
            font-size: 15px !important;
            min-height: 48px !important;
            resize: none !important;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06) !important;
            transition: all 0.3s ease !important;
        }
        
        .stChatInput > div > div > textarea:focus {
            border-color: #10b981 !important;
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15), 0 4px 6px -1px rgba(16, 185, 129, 0.1) !important;
            outline: none !important;
            background-color: #f0fdf4 !important;
        }
        
        /* Send button with gradient and hover effect */
        .stChatInput > div > div > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
            border: none !important;
            border-radius: 24px !important;
            width: 48px !important;
            height: 48px !important;
            color: white !important;
            box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.4), 0 2px 4px -1px rgba(16, 185, 129, 0.3) !important;
            transition: all 0.3s ease !important;
        }
        
        .stChatInput > div > div > button:hover {
            background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
            transform: scale(1.05) !important;
            box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.5), 0 4px 6px -2px rgba(16, 185, 129, 0.4) !important;
        }
        
        .stChatInput > div > div > button:active {
            transform: scale(0.95) !important;
        }
        
        /* Add padding to main area for fixed input */
        .main {
            padding-bottom: 100px !important;
            padding-top: 0 !important;
        }
        
        /* Nuclear option - remove ALL possible spacing */
        .stApp div:not(.header):not(.input-area):not(.input-container) {
            margin-top: 0 !important;
        }
        
        /* Target all possible Streamlit containers */
        div[data-testid="stBlock"],
        div[data-testid="element-container"],
        div[data-testid="stVerticalBlock"],
        div[data-testid="stForm"],
        .element-container {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Remove gap from the main content area */
        .main > div {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Remove spacing from the first element after header */
        .header ~ * {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        .header + * {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        /* Extreme measures to eliminate any gap */
        .stApp > div:nth-child(2) {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        .stApp > div:nth-child(2) > div {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        .stApp > div:nth-child(2) > div > div {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        /* Suggestion pills hover effect */
        span[style*="background: linear-gradient(135deg, #ecfdf5"] {
            transition: all 0.2s ease;
        }
        
        span[style*="background: linear-gradient(135deg, #ecfdf5"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3);
        }
        
        /* Target the exact first chat message */
        .chat-area [data-testid="stChatMessage"]:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        .chat-area [data-testid="stChatMessage"]:first-child > div {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        /* Make the chat area stick to header */
        .header + div .chat-area {
            margin-top: -2px !important;
            padding-top: 0 !important;
        }
        
        /* Ultra-aggressive spacing removal */
        .stApp [data-testid="stAppViewContainer"] {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        .stApp [data-testid="stAppViewContainer"] > div {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        .stApp [data-testid="stAppViewContainer"] > div > div {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        .stApp .main {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        .stApp .main > div {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        .stApp .main .block-container {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        /* Remove default Streamlit spacing completely */
        .stApp section[data-testid="stSidebar"] ~ div,
        .stApp section[data-testid="stSidebar"] ~ div > div,
        .stApp section[data-testid="stSidebar"] ~ div > div > div {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        /* Force the chat area to start immediately */
        .stApp > div:not(.header) {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        .stApp > div:not(.header) > div {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        .stApp > div:not(.header) > div > div {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Load logo image for header
    logo_img_src = None
    try:
        logo_path = os.path.join("Assets", "logo.png")
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as img_file:
                logo_img_base64 = base64.b64encode(img_file.read()).decode()
                logo_img_src = f"data:image/png;base64,{logo_img_base64}"
    except Exception as e:
        print(f"Could not load logo image: {e}")
        logo_img_src = None

    # Page Header with logo image
    if logo_img_src:
        logo_html = f'<img src="{logo_img_src}" style="width: 24px; height: 24px; border-radius: 4px; object-fit: cover;">'
    else:
        logo_html = '<div class="logo-icon">FS</div>'
    
    st.markdown(f"""
    <div class="header">
        <div class="logo">
            {logo_html}
            FarmSmart
        </div>
        <div class="center-title">
            <div class="main-title">GrowMate</div>
            <div class="subtitle">• Your Hydroponic Farming Expert •</div>
        </div>
        <div class="status">
            <div class="status-dot"></div>
            Online
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize model and chat
    ensure_model_loaded()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load and encode loader image for bot avatar
    loader_img_src = None
    try:
        loader_path = os.path.join("Assets", "loader.png")
        if os.path.exists(loader_path):
            with open(loader_path, "rb") as img_file:
                loader_img_base64 = base64.b64encode(img_file.read()).decode()
                loader_img_src = f"data:image/png;base64,{loader_img_base64}"
    except Exception as e:
        print(f"Could not load loader image: {e}")
        loader_img_src = None
    
    # Enhanced welcome message with avatar and quick questions
    if loader_img_src:
        avatar_html = f'<img src="{loader_img_src}" style="width: 40px; height: 40px; border-radius: 50%; object-fit: cover; border: 2px solid #10b981; box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3);">'
    else:
        avatar_html = '<div style="width: 40px; height: 40px; background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid #10b981; box-shadow: 0 4px 6px -1px rgba(249, 115, 22, 0.3);"><span style="color: white; font-size: 16px; font-weight: bold;">GM</span></div>'
    
    # Get current time
    current_time = datetime.now().strftime("%I:%M %p")
    
    # Welcome message - split into two separate markdown blocks to avoid rendering issues
    st.markdown(f"""
    <div style="max-width: 800px; margin: 0 auto; padding: 32px 20px 20px 20px; animation: fadeInUp 0.6s ease-out;">
        <div style="display: flex; align-items: flex-start; margin-bottom: 24px;">
            <div style="margin-right: 16px; flex-shrink: 0;">
                {avatar_html}
            </div>
            <div>
                <div style="background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%); border-radius: 20px 20px 20px 6px; padding: 16px 20px; color: #374151; max-width: 85%; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); border: 1px solid #e5e7eb;">
                    <p style="margin: 0 0 12px 0; line-height: 1.6; font-size: 15px; font-weight: 500;">Welcome! I'm <span style="color: #10b981; font-weight: 700;">GrowMate</span>, your hydroponic farming assistant powered by AI.</p>
                    <p style="margin: 0; line-height: 1.5; font-size: 14px; color: #6b7280;">Ask me anything about hydroponic systems, nutrient solutions, plant care, or getting started with hydroponics. I'm here to help you grow!</p>
                    <div style="font-size: 11px; color: #9ca3af; margin-top: 8px; font-weight: 500;">{current_time}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Suggestion pills in a separate block
    st.markdown("""
    <div style="max-width: 800px; margin: -8px auto 0 auto; padding: 0 20px 20px 76px;">
        <p style="font-size: 12px; color: #6b7280; margin-bottom: 8px; font-weight: 600;">Try asking:</p>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
            <span style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); color: #047857; padding: 8px 14px; border-radius: 16px; font-size: 13px; border: 1px solid #a7f3d0; display: inline-block; font-weight: 500;">
                What is hydroponic farming?
            </span>
            <span style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); color: #047857; padding: 8px 14px; border-radius: 16px; font-size: 13px; border: 1px solid #a7f3d0; display: inline-block; font-weight: 500;">
                How do I maintain pH levels?
            </span>
            <span style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); color: #047857; padding: 8px 14px; border-radius: 16px; font-size: 13px; border: 1px solid #a7f3d0; display: inline-block; font-weight: 500;">
                Best crops for beginners?
            </span>
            <span style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); color: #047857; padding: 8px 14px; border-radius: 16px; font-size: 13px; border: 1px solid #a7f3d0; display: inline-block; font-weight: 500;">
                NFT system setup guide
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display all conversation messages using pure HTML with enhanced styling
    if st.session_state.messages:
        messages_html = ""
        for message in st.session_state.messages:
            if message["role"] == "user":
                # User message - right aligned, green gradient
                messages_html += f"""
                <div style="max-width: 800px; margin: 0 auto; padding: 0 20px; margin-bottom: 12px; margin-top: 24px; animation: fadeInRight 0.4s ease-out;">
                    <div style="display: flex; justify-content: flex-end;">
                        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 18px 18px 4px 18px; padding: 14px 18px; color: white; max-width: 75%; box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3), 0 2px 4px -1px rgba(16, 185, 129, 0.2); transition: transform 0.2s ease;" onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
                            <p style="margin: 0; line-height: 1.5; font-size: 15px;">{message["content"]}</p>
                            <div style="font-size: 11px; color: #d1fae5; margin-top: 6px; text-align: right; font-weight: 500;">{current_time}</div>
                        </div>
                    </div>
                </div>
                """
            else:
                # Assistant message - left aligned with enhanced avatar
                if loader_img_src:
                    avatar_img = f'<img src="{loader_img_src}" style="width: 36px; height: 36px; border-radius: 50%; object-fit: cover; border: 2px solid #10b981; box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);">'
                else:
                    avatar_img = '<div style="width: 36px; height: 36px; background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid #10b981; box-shadow: 0 2px 4px rgba(249, 115, 22, 0.2);"><span style="color: white; font-size: 16px; font-weight: bold;">GM</span></div>'
                
                messages_html += f"""
                <div style="max-width: 800px; margin: 0 auto; padding: 0 20px; margin-bottom: 24px; margin-top: 12px; animation: fadeInLeft 0.4s ease-out;">
                    <div style="display: flex; align-items: flex-start;">
                        <div style="margin-right: 14px; flex-shrink: 0;">
                            {avatar_img}
                        </div>
                        <div style="background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%); border-radius: 18px 18px 18px 4px; padding: 14px 18px; color: #374151; max-width: 75%; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); border: 1px solid #e5e7eb; transition: transform 0.2s ease;" onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
                            <p style="margin: 0; line-height: 1.5; font-size: 15px;">{message["content"]}</p>
                            <div style="font-size: 11px; color: #9ca3af; margin-top: 6px; font-weight: 500;">{current_time}</div>
                        </div>
                    </div>
                </div>
                """
        
        st.markdown(messages_html, unsafe_allow_html=True)

    # Fixed input area
    st.markdown("""
    <div class="input-area">
        <div class="input-container">
    """, unsafe_allow_html=True)
    
    # Chat input
    user_prompt = st.chat_input("Ask about hydroponics, nutrients, systems, plants...")
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Handle input with enhanced styling
    if user_prompt:
        # Add user message to session state and display immediately
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        # Display the user's question immediately with enhanced styling
        user_message_html = f"""
        <div style="max-width: 800px; margin: 0 auto; padding: 0 20px; margin-bottom: 12px; margin-top: 24px; animation: fadeInRight 0.4s ease-out;">
            <div style="display: flex; justify-content: flex-end;">
                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 18px 18px 4px 18px; padding: 14px 18px; color: white; max-width: 75%; box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3), 0 2px 4px -1px rgba(16, 185, 129, 0.2);">
                    <p style="margin: 0; line-height: 1.5; font-size: 15px;">{user_prompt}</p>
                    <div style="font-size: 11px; color: #d1fae5; margin-top: 6px; text-align: right; font-weight: 500;">{current_time}</div>
                </div>
            </div>
        </div>
        """
        st.markdown(user_message_html, unsafe_allow_html=True)
        
        # Show enhanced thinking indicator with animation
        with st.spinner("GrowMate is thinking..."):
            tokenizer = st.session_state.get("tokenizer")
            model = st.session_state.get("model")
            model_type = st.session_state.get("model_type", "Base FLAN-T5")
            device = st.session_state.get("device", torch.device('cpu') if 'torch' in globals() else None)
            
            response = generate_response(user_prompt, tokenizer, model, model_type, device)
        
        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display the assistant's response immediately with enhanced styling
        if loader_img_src:
            avatar_display = f'<img src="{loader_img_src}" style="width: 36px; height: 36px; border-radius: 50%; object-fit: cover; border: 2px solid #10b981; box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);">'
        else:
            avatar_display = '<div style="width: 36px; height: 36px; background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid #10b981; box-shadow: 0 2px 4px rgba(249, 115, 22, 0.2);"><span style="color: white; font-size: 16px; font-weight: bold;">GM</span></div>'
        
        assistant_message_html = f"""
        <div style="max-width: 800px; margin: 0 auto; padding: 0 20px; margin-bottom: 24px; margin-top: 12px; animation: fadeInLeft 0.4s ease-out;">
            <div style="display: flex; align-items: flex-start;">
                <div style="margin-right: 14px; flex-shrink: 0;">
                    {avatar_display}
                </div>
                <div style="background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%); border-radius: 18px 18px 18px 4px; padding: 14px 18px; color: #374151; max-width: 75%; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); border: 1px solid #e5e7eb;">
                    <p style="margin: 0; line-height: 1.5; font-size: 15px;">{response}</p>
                    <div style="font-size: 11px; color: #9ca3af; margin-top: 6px; font-weight: 500;">{current_time}</div>
                </div>
            </div>
        </div>
        """
        st.markdown(assistant_message_html, unsafe_allow_html=True)
if __name__ == "__main__":
    main()
