import streamlit as st
import os
import json
from datetime import datetime
from PIL import Image
from io import BytesIO
import base64

# --- Backend Model Dependencies ---
try:
    # def main():
    # Load loader image for favicon
    favicon_path = os.path.join("Assets", "loader.png")
    if os.path.exists(favicon_path):
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
            page_icon="🌱",
            layout="wide",
            initial_sidebar_state="collapsed"
        ) 
    # to import necessary libraries for the LLM
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch
except ImportError:
    # Define placeholder classes/functions if dependencies are missing for running the Streamlit UI
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

# --- Configuration for LLM ---
class Config:
    MODEL_NAME = "google/flan-t5-base"
    MAX_INPUT_LENGTH = 512
    MAX_OUTPUT_LENGTH = 256
    TEMPERATURE = 0.7
    NUM_BEAMS = 4

# --- Utility: Encode Leaf Icon as Base64 for CSS ---
# Using a simple green leaf SVG for the logo in the custom header
LEAF_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-leaf"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 16 12 16 12s-3.2 1.4-8.7 2.8c-1.6 4.7-5.2 6.5-8.7 2.8s-.2 1.4-8.7 2.8c-1.6 4.7-5.2 6.5-8.7 2.8c-1.6 4.7-5.2 6.5-8.7 2.8c-1.6 4.7-5.2 6.5-8.7 2.8z"/></path><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 16 12 16 12s-3.2 1.4-8.7 2.8c-1.6 4.7-5.2 6.5-8.7 2.8z"/></path></svg>
"""

def svg_to_base64(svg_string):
    """Encodes SVG string to base64 data URI for CSS use."""
    return f"data:image/svg+xml;base64,{base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')}"

# --- LLM Loading and Generation Functions (Provided by User) ---

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
                    # Optimize model loading for deployment
                    model = T5ForConditionalGeneration.from_pretrained(
                        trained_model_dir,
                        torch_dtype=torch.float16,  # Use float16 for smaller size
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
    st.set_page_config(
        page_title="FarmSmart - GrowMate",
        page_icon="�",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Clean, minimal CSS exactly matching the screenshot
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
        
        .stApp {
            background-color: white;
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
        
        /* Header styling exactly like screenshot */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 20px;
            background-color: white;
            border-bottom: 1px solid #e5e7eb;
            margin: 0;
            position: relative;
            top: 0;
            z-index: 1000;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
            font-size: 16px;
            color: #374151;
        }
        
        .logo-icon {
            width: 24px;
            height: 24px;
            background-color: #10b981;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
        }
        
        .center-title {
            text-align: center;
            flex-grow: 1;
        }
        
        .main-title {
            font-size: 18px;
            font-weight: 600;
            color: #10b981;
            margin: 0;
        }
        
        .subtitle {
            font-size: 12px;
            color: #6b7280;
            margin: 0;
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: #10b981;
        }
        
        .status-dot {
            width: 6px;
            height: 6px;
            background-color: #10b981;
            border-radius: 50%;
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
        
        /* Message bubbles exactly like screenshot */
        [data-testid="stChatMessage"] {
            margin-bottom: 16px;
        }
        
        /* First message - start immediately */
        [data-testid="stChatMessage"]:first-child {
            margin-top: 0px;
        }
        
        /* Bot messages - light gray, left aligned */
        [data-testid="stChatMessage"][data-testid="assistant"] [data-testid="stMarkdownContainer"],
        [data-testid="stChatMessage"]:not([data-testid="user"]) [data-testid="stMarkdownContainer"] {
            background-color: #f3f4f6 !important;
            border-radius: 16px 16px 16px 4px !important;
            padding: 12px 16px !important;
            color: #374151 !important;
            margin: 0 !important;
            max-width: 80% !important;
            display: inline-block !important;
        }
        
        /* User messages - green background, right aligned */
        [data-testid="stChatMessage"][data-testid="user"] [data-testid="stMarkdownContainer"] {
            background-color: #10b981 !important;
            border-radius: 16px 16px 4px 16px !important;
            padding: 12px 16px !important;
            color: white !important;
            margin: 0 !important;
            max-width: 80% !important;
            display: inline-block !important;
            float: right !important;
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
        
        /* Input area at bottom */
        .input-area {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: white;
            border-top: 1px solid #e5e7eb;
            padding: 16px 20px;
            z-index: 1000;
        }
        
        .input-container {
            max-width: 800px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        /* Chat input styling */
        .stChatInput {
            flex: 1;
        }
        
        .stChatInput > div > div > textarea {
            border-radius: 20px !important;
            border: 1px solid #d1d5db !important;
            background-color: #f9fafb !important;
            padding: 10px 16px !important;
            font-size: 14px !important;
            min-height: 40px !important;
            resize: none !important;
        }
        
        .stChatInput > div > div > textarea:focus {
            border-color: #10b981 !important;
            box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.1) !important;
            outline: none !important;
        }
        
        /* Send button */
        .stChatInput > div > div > button {
            background-color: #374151 !important;
            border: none !important;
            border-radius: 20px !important;
            width: 40px !important;
            height: 40px !important;
            color: white !important;
        }
        
        .stChatInput > div > div > button:hover {
            background-color: #059669 !important;
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
        logo_html = '<div class="logo-icon">🌿</div>'
    
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
    
    # Welcome message directly after header (with comfortable spacing)
    if loader_img_src:
        avatar_html = f'<img src="{loader_img_src}" style="width: 32px; height: 32px; border-radius: 50%; object-fit: cover; border: 1px solid #e5e7eb;">'
    else:
        avatar_html = '<div style="width: 32px; height: 32px; background-color: #f97316; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 1px solid #e5e7eb;"><span style="color: white; font-size: 16px;">🤖</span></div>'
    
    st.markdown(f"""
    <div style="max-width: 800px; margin: 0 auto; padding: 24px 20px 0 20px;">
        <div style="display: flex; align-items: flex-start; margin-bottom: 16px;">
            <div style="margin-right: 12px; flex-shrink: 0;">
                {avatar_html}
            </div>
            <div style="background-color: #f3f4f6; border-radius: 16px 16px 16px 4px; padding: 12px 16px; color: #374151; max-width: 80%;">
                <p style="margin: 0; line-height: 1.4; font-size: 14px;">Welcome! I'm GrowMate, your hydroponic farming assistant. Ask me anything about hydroponic systems, nutrient solutions, plant care, or getting started with hydroponics.</p>
                <div style="font-size: 11px; color: #9ca3af; margin-top: 4px;">02:46 PM</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display all conversation messages using pure HTML (no Streamlit chat components)
    if st.session_state.messages:
        messages_html = ""
        for message in st.session_state.messages:
            if message["role"] == "user":
                # User message - right aligned, green background
                messages_html += f"""
                <div style="max-width: 800px; margin: 0 auto; padding: 0 20px; margin-bottom: 8px; margin-top: 20px;">
                    <div style="display: flex; justify-content: flex-end;">
                        <div style="background-color: #10b981; border-radius: 16px 16px 4px 16px; padding: 12px 16px; color: white; max-width: 80%;">
                            <p style="margin: 0; line-height: 1.4; font-size: 14px;">{message["content"]}</p>
                            <div style="font-size: 11px; color: #d1fae5; margin-top: 4px; text-align: right;">02:46 PM</div>
                        </div>
                    </div>
                </div>
                """
            else:
                # Assistant message - left aligned, light background
                if loader_img_src:
                    avatar_img = f'<img src="{loader_img_src}" style="width: 32px; height: 32px; border-radius: 50%; object-fit: cover; border: 1px solid #e5e7eb;">'
                else:
                    avatar_img = '<div style="width: 32px; height: 32px; background-color: #f97316; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 1px solid #e5e7eb;"><span style="color: white; font-size: 16px;">🤖</span></div>'
                
                messages_html += f"""
                <div style="max-width: 800px; margin: 0 auto; padding: 0 20px; margin-bottom: 20px; margin-top: 8px;">
                    <div style="display: flex; align-items: flex-start;">
                        <div style="margin-right: 12px; flex-shrink: 0;">
                            {avatar_img}
                        </div>
                        <div style="background-color: #f3f4f6; border-radius: 16px 16px 16px 4px; padding: 12px 16px; color: #374151; max-width: 80%;">
                            <p style="margin: 0; line-height: 1.4; font-size: 14px;">{message["content"]}</p>
                            <div style="font-size: 11px; color: #9ca3af; margin-top: 4px;">02:46 PM</div>
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

    # Handle input
    if user_prompt:
        # Add user message to session state and display immediately
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        # Display the user's question immediately with proper formatting
        user_message_html = f"""
        <div style="max-width: 800px; margin: 0 auto; padding: 0 20px; margin-bottom: 8px; margin-top: 20px;">
            <div style="display: flex; justify-content: flex-end;">
                <div style="background-color: #10b981; border-radius: 16px 16px 4px 16px; padding: 12px 16px; color: white; max-width: 80%;">
                    <p style="margin: 0; line-height: 1.4; font-size: 14px;">{user_prompt}</p>
                    <div style="font-size: 11px; color: #d1fae5; margin-top: 4px; text-align: right;">02:46 PM</div>
                </div>
            </div>
        </div>
        """
        st.markdown(user_message_html, unsafe_allow_html=True)
        
        # Show thinking indicator
        with st.spinner("GrowMate is thinking..."):
            tokenizer = st.session_state.get("tokenizer")
            model = st.session_state.get("model")
            model_type = st.session_state.get("model_type", "Base FLAN-T5")
            device = st.session_state.get("device", torch.device('cpu') if 'torch' in globals() else None)
            
            response = generate_response(user_prompt, tokenizer, model, model_type, device)
        
        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display the assistant's response immediately with proper formatting
        if loader_img_src:
            avatar_display = f'<img src="{loader_img_src}" style="width: 32px; height: 32px; border-radius: 50%; object-fit: cover; border: 1px solid #e5e7eb;">'
        else:
            avatar_display = '<div style="width: 32px; height: 32px; background-color: #f97316; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 1px solid #e5e7eb;"><span style="color: white; font-size: 16px;">🤖</span></div>'
        
        assistant_message_html = f"""
        <div style="max-width: 800px; margin: 0 auto; padding: 0 20px; margin-bottom: 20px; margin-top: 8px;">
            <div style="display: flex; align-items: flex-start;">
                <div style="margin-right: 12px; flex-shrink: 0;">
                    {avatar_display}
                </div>
                <div style="background-color: #f3f4f6; border-radius: 16px 16px 16px 4px; padding: 12px 16px; color: #374151; max-width: 80%;">
                    <p style="margin: 0; line-height: 1.4; font-size: 14px;">{response}</p>
                    <div style="font-size: 11px; color: #9ca3af; margin-top: 4px;">02:46 PM</div>
                </div>
            </div>
        </div>
        """
        st.markdown(assistant_message_html, unsafe_allow_html=True)
if __name__ == "__main__":
    main()
