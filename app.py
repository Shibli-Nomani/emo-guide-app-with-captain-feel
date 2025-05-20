# app.py
import gradio as gr
import cv2
import json
import random
import tempfile
import pandas as pd
import plotly.graph_objects as go
from deepface import DeepFace
from TTS.api import TTS
from llama_cpp import Llama
from utils.emotion_analysis import analyze_emotion_and_display, analyze_full_info, create_hierarchical_tree, plot_emotion_confidence
from utils.nlp_advice import get_default_questions, get_advice_from_json, chat, respond
from utils.audio_processing import generate_welcome_audio

# Load LLaMA 2 model (GGUF format)
llm = Llama(
    model_path="llama-2-7b-chat.Q4_K_M.gguf",  # Download and include in repo or use Hugging Face model hub
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0  # CPU-only for Hugging Face Spaces free tier
)

# Load TTS model
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)
speaker = "p226"

# Load JSON advice data
with open("emontions_advice.json", "r") as f:
    advice_data = json.load(f)

# Generate welcome audio
welcome_audio_path = generate_welcome_audio()

# Gradio app logic
def gradio_app(image):
    result_img = analyze_emotion_and_display(image)
    df = analyze_full_info(image)
    emotion = df['dominant_emotion'][0]
    markdown_text = f"### ✅ Captain Feels: I sense you're feeling **{emotion.upper()}**."
    
    questions = get_default_questions(emotion)
    chatbot_start = [["", f"Captain Feels 🤖: Q1: {questions[0]}"]]
    
    fig1 = create_hierarchical_tree(df)
    fig2 = plot_emotion_confidence(df)
    
    return (
        result_img, fig1, fig2,
        markdown_text, chatbot_start, emotion,
        questions, 1
    )

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 😎 EmoGuide: Emotion-Aware Conversations with Captain Feels")
    gr.Markdown("## 🚩 Play the Audio and Start the Journey with Patience")
    audio_player = gr.Audio(value=welcome_audio_path, autoplay=True, label="🔊 Captain Feels Says...")

    with gr.Row():
        image_input = gr.Image(type="numpy", label="📸 Upload or Capture Image")
        result_image = gr.Image(type="numpy", label="🧠 Emotion Detection Output")

    run_button = gr.Button("Analyze Emotion 🧠")

    fig1_plot = gr.Plot(label="🌳 Personal Info Treemap")
    fig2_plot = gr.Plot(label="📊 Emotion Confidence Plot")
    advice_output = gr.Markdown("### Captain Feels: Waiting for image...")
    gr.Markdown("##### 🚩 To achieve better results, please write well-structured prompts with explanations.")
    chatbot = gr.Chatbot(label="Captain Feels 🤖", value=[])
    user_input = gr.Textbox(label="💬 Your Message", placeholder="Type your response or 'exit' to end...")
    send_button = gr.Button("Send")

    # States
    emotion_state = gr.State("neutral")
    questions_state = gr.State([])
    question_index = gr.State(0)

    # Image processing
    run_button.click(
        fn=gradio_app,
        inputs=[image_input],
        outputs=[
            result_image, fig1_plot, fig2_plot,
            advice_output, chatbot,
            emotion_state, questions_state, question_index
        ]
    )

    # Chat interaction
    send_button.click(
        fn=respond,
        inputs=[
            emotion_state, user_input, chatbot,
            questions_state, question_index
        ],
        outputs=[
            chatbot, user_input, questions_state, question_index
        ]
    )

if __name__ == "__main__":
    app.launch()
