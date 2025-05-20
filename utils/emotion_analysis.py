# utils/emotion_analysis.py
import cv2
import pandas as pd
import plotly.graph_objects as go
from deepface import DeepFace

def analyze_emotion_and_display(image):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, img_bgr)
    emotions = DeepFace.analyze(img_path=temp_path, actions=['emotion'])
    padding = 50
    img_rgb_padded = cv2.copyMakeBorder(image, padding, padding, padding, padding, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

    for face in emotions:
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        dominant_emotion = face['dominant_emotion']
        emotion_percentage = face['emotion'][dominant_emotion]

        text = f"{dominant_emotion}: {emotion_percentage:.2f}%"
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_width, text_height = text_size
        pad = 20
        x += padding
        y += padding
        rect_width = max(w, text_width) + pad * 2
        rect_x1 = x - pad
        rect_x2 = rect_x1 + rect_width
        rect_y1 = y - text_height - pad - 10
        rect_y2 = rect_y1 + text_height + pad

        cv2.rectangle(img_rgb_padded, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        cv2.rectangle(img_rgb_padded, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 0), 4)
        text_x = rect_x1 + (rect_width - text_width) // 2
        text_y = rect_y1 + (text_height + pad) // 2 + baseline
        cv2.putText(img_rgb_padded, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        face_rect_x1 = x - pad
        face_rect_x2 = max(x + w + pad, rect_x2)
        cv2.rectangle(img_rgb_padded, (face_rect_x1, y), (face_rect_x2, y + h), (255, 0, 0), 4)

    return img_rgb_padded

def analyze_full_info(image):
    temp_path = "temp_image.jpg"
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_path, image_bgr)
    analysis = DeepFace.analyze(img_path=temp_path, actions=['emotion', 'age', 'gender'])
    df = pd.json_normalize(analysis)
    return df

def create_hierarchical_tree(df):
    labels = [
        "Person Info",
        f"CURRENT EMOTION: {df['dominant_emotion'][0].upper()}",
        f"FACE CONFIDENCE: {df['face_confidence'][0] * 100:.2f}%",
        f"APPROXIMATE AGE: {df['age'][0]}",
        f"GENDER: {df['dominant_gender'][0].upper()}"
    ]
    parents = ["", "Person Info", "Person Info", "Person Info", "Person Info"]

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        marker=dict(colors=[0, 1, 2, 3, 4], colorscale='Sunset'),
        textinfo="label+value",
        textfont=dict(size=16, family="Arial", color="black"),
        insidetextfont=dict(size=16, color="black"),
        textposition="middle center"
    ))

    fig.update_layout(title="Emotion And Personal Info From Photo")
    return fig

def plot_emotion_confidence(df):
    df_filter = df[['emotion.angry', 'emotion.disgust', 'emotion.fear', 'emotion.happy', 'emotion.sad', 'emotion.surprise', 'emotion.neutral']]
    df_filter.columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    emotion_to_emoji = {
        'angry': '😠', 'disgust': '😷', 'fear': '😨',
        'happy': '😊', 'sad': '😢', 'surprise': '😲', 'neutral': '😐'
    }
    emotion_to_color = {
        'angry': '#B22222', 'disgust': '#8B4513', 'fear': '#800080',
        'happy': '#006400', 'sad': '#1E90FF', 'surprise': '#FF8C00', 'neutral': '#696969'
    }

    sorted_columns = df_filter.iloc[0].sort_values(ascending=False)
    fig = go.Figure()
    for emo in sorted_columns.index:
        fig.add_trace(go.Bar(
            x=[sorted_columns[emo]],
            y=[f"{emotion_to_emoji[emo]} {emo}"],
            orientation='h',
            marker=dict(color=emotion_to_color[emo]),
            name=emo,
            text=[f"{sorted_columns[emo]:.2f}%"],
            textposition='outside'
        ))

    fig.update_layout(
        title="Emotion Confidence Levels with Emojis",
        xaxis_title="Confidence Level",
        yaxis_title="Emotion",
        barmode='stack',
        height=400
    )
    fig.update_xaxes(range=[0, 110])
    return fig
