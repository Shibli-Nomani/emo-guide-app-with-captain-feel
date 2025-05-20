# utils/audio_processing.py
import tempfile
from TTS.api import TTS

def generate_welcome_audio():
    welcome_text = (
        "Hello! I’m Captain Feels — your virtual Emo Bot and Emotion Assistant. "
        "Please be patient as we go through the process. First, you’ll hear a short audio message. "
        "Then, upload or drag and drop your image for analysis. Finally, I’ll guide you with personalized advice and a few interactive questions."
    )
    tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)
    _, audio_path = tempfile.mkstemp(suffix=".wav")
    tts.tts_to_file(text=welcome_text, speaker="p226", file_path=audio_path)
    return audio_path
