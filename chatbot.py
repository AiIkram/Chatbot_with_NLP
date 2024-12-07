# -*- coding: utf-8 -*-

import nltk
import string
import streamlit as st
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

file_path = 'mental_health_chatbot.txt'
def load_and_preprocess(file_path):
    """
    Load and preprocess a text file containing Q&A pairs.

    Args:
        file_path (str): Path to the text file.

    Returns:
        list: A list of tuples, where each tuple contains a question and its corresponding answer.
    """
    # Load the text file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().replace('\n', ' ')  # Replace newlines with spaces for consistency

    # Split the text into sentences
    sentences = sent_tokenize(data)

    # Initialize variables for storing Q&A pairs
    qa_pairs = []
    current_q, current_a = None, None

    # Process sentences to identify Q&A pairs
    for sentence in sentences:
        if '?' in sentence:  # Identify questions
            if current_q and current_a:  # Save the previous Q&A pair if exists
                qa_pairs.append((current_q, current_a))
            current_q = sentence  # Update current question
            current_a = None  # Reset current answer
        else:
            current_a = sentence  # Identify the answer

    # Add the last Q&A pair if valid
    if current_q and current_a:
        qa_pairs.append((current_q, current_a))

    return qa_pairs

def preprocess(text):
    """
    Preprocess user input text for NLP tasks.

    Args:
        text (str): The input text from the user.

    Returns:
        list: A list of preprocessed tokens.
    """
    # Tokenize text into words and convert to lowercase
    tokens = word_tokenize(text.lower())

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))  # Load stopwords
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]

    # Lemmatize tokens to their base forms
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

def find_best_match(user_input, qa_pairs):
    """
    Find the best matching question-answer pair based on Jaccard similarity.

    Args:
        user_input (str): The input text from the user.
        qa_pairs (list of tuple): A list of question-answer pairs, where each item is a tuple (question, answer).

    Returns:
        tuple: The best matching question-answer pair (question, answer) based on highest similarity.
    """
    processed_input = preprocess(user_input)
    best_match = None
    max_similarity = 0

    for question, answer in qa_pairs:
        processed_question = preprocess(question)
        # Calculate Jaccard similarity
        similarity = len(set(processed_input).intersection(set(processed_question))) / \
                     len(set(processed_input).union(set(processed_question)))
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = (question, answer)

    return best_match if best_match else "I'm sorry, I don't have an answer to that."

# Streamlit UI

st.title("Mental Health Chatbot")
st.write("This chatbot provides answers to mental health-related questions.")

# Load and preprocess Q&A dataset

qa_pairs = load_and_preprocess(file_path)

# User input
user_input = st.text_input("Ask me anything about mental health:")
if user_input:
    # Generate chatbot response
    response = find_best_match(user_input, qa_pairs)
    st.write(f"Chatbot: {response}")
