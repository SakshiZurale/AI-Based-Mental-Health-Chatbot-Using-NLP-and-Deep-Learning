# chatbot.py

import json
import random
import nltk
import pickle
import streamlit as st
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher

# ------------------------------
# NLTK Setup
# ------------------------------
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# ------------------------------
# Load intents and vocabulary
# ------------------------------
with open('intents.json') as f:
    intents = json.load(f)

words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# ------------------------------
# NLP Utilities
# ------------------------------
def clean_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in tokens]

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def predict_intent(sentence, context=None, threshold=0.6):
    """
    Predict intent with optional context.
    If a context exists, prioritize intents matching context.
    """
    sentence_words = clean_sentence(sentence)
    best_match = None
    max_score = 0

    for intent in intents['intents']:
        # Skip intents not matching context if context is set
        if context and 'context_filter' in intent and intent['context_filter'] != context:
            continue

        for pattern in intent['patterns']:
            pattern_words = clean_sentence(pattern)
            score = similar(" ".join(sentence_words), " ".join(pattern_words))
            if score > max_score:
                max_score = score
                best_match = intent['tag']

    if max_score < threshold:
        return None
    return best_match

def get_response(intent_tag):
    if intent_tag:
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                response = random.choice(intent['responses'])
                # Optional: return context for next turn
                context_set = intent.get('context_set', None)
                return response, context_set
    return "Sorry, I didn’t understand that.", None

# ------------------------------
# Chatbot logic with context
# ------------------------------
def chatbot_response(msg, context=None):
    intent_tag = predict_intent(msg, context)
    response, context_set = get_response(intent_tag)
    return response, context_set

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="💬 AI Chatbot", page_icon="💬", layout="centered")
st.title("💬 Your Intelligent Chat Companion")
st.caption("Bridging Conversations, One Message at a Time")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("context", None)

# Display chat history
for chat in st.session_state.messages:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Input box
user_input = st.chat_input("Type your message here...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Bot is thinking..."):
        response, context_set = chatbot_response(user_input, st.session_state.context)
        st.session_state.context = context_set  # update context for next turn
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
