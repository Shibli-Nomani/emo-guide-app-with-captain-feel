# utils/nlp_advice.py
import random

def get_default_questions(emotion, advice_data):
    questions = [entry["question"] for entry in advice_data if entry["emotion"] == emotion]
    return questions[:5] if questions else ["Can you share how you're feeling today?"]

def get_advice_from_json(emotion, user_response, advice_data, llm):
    entries = [entry for entry in advice_data if entry["emotion"] == emotion]
    if not entries:
        return "Sorry, I don’t have advice for that feeling yet."
    entry = random.choice(entries)
    advice = random.choice(entry["advice"])
    prompt = (
        f"User feels {emotion} and said: '{user_response}'.\n"
        f"Advice: \"{advice}\"\n"
        f"Rewrite this as a single empathetic sentence directly addressing the user's situation. "
        f"Respond only with the rephrased line in quotes, no explanation or questions."
    )
    try:
        response = llm(prompt, max_new_tokens=60, temperature=0.7)
        return response.strip().strip('"')
    except Exception as e:
        return f"Error generating advice: {e}"

def chat(emotion, user_input, history, advice_data, llm):
    if user_input.lower() in ["exit", "goodbye", "stop"]:
        final_msg = (
            "Captain Feels🤖: Thank you for exploring. If you liked it, share & like my LinkedIn post 🌟\n\n"
            "🤖 Keep shining. Take care 🌸"
        )
        return history + [[f"You: {user_input}", final_msg]], ""
    advice = get_advice_from_json(emotion, user_input, advice_data, llm)
    new_pair = [f"You: {user_input}", f"💡 Advice: {advice}"]
    return history + [new_pair], ""

def respond(emotion, user_text, chat_hist, questions, q_idx, advice_data, llm):
    if user_text.lower() in ["exit", "goodbye", "stop"]:
        return chat(emotion, user_text, chat_hist, advice_data, llm) + (questions, q_idx)

    updated_chat, _ = chat(emotion, user_text, chat_hist, advice_data, llm)

    if q_idx < len(questions):
        next_q = questions[q_idx]
        updated_chat.append([f"Captain Feels 🤖: Q{q_idx + 1}: {next_q}", ""])
        q_idx += 1
    else:
        updated_chat.append(["Captain Feels 🤖: You’ve finally completed the journey. You’re doing your best 🌈.\n\n"
                             "If you enjoyed this experience, please like and share my LinkedIn post 🌟. 🤖 Keep shining, take care 🌸, and thank you for sharing! 🌈", ""])

    return updated_chat, "", questions, q_idx
