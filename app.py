import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import base64

# Streamlit page configuration
st.set_page_config(page_title="NER Model", layout="wide")

# Function to get base64 of binary file for background image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set background image
def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image
set_background('2148999166.jpg')

# Maximum sentence length and number of words
maxlen = 110
max_words = 36000

# Load the dataset and tokenizer
ready_data = pd.read_csv('ner.csv')
X = list(ready_data['Sentence'])

# Tokenize sentences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)

# Define the id2word and id2tag mappings
id2tag = {0: 'O', 1: 'B-geo', 2: 'B-gpe', 3: 'B-per', 4: 'I-geo',
          5: 'B-org', 6: 'I-org', 7: 'B-tim', 8: 'B-art', 9: 'I-art',
          10: 'I-per', 11: 'I-gpe', 12: 'I-tim', 13: 'B-nat', 14: 'B-eve',
          15: 'I-eve', 16: 'I-nat'}

# Function to create id2word mapping
def id2word():
    word_index = tokenizer.word_index
    id2word = {v: k for k, v in word_index.items()}
    return id2word

# Preprocessing function for text input
def preprocessingg(text):
    sequences = tokenizer.texts_to_sequences(text)
    text = pad_sequences(sequences, maxlen=maxlen, padding='post')
    return np.asarray(text)

# Function to make predictions
def make_prediction(model, preprocessed_sentence, id2word, id2tag):
    preprocessed_sentence = preprocessed_sentence.reshape((1, maxlen))
    sentence = preprocessed_sentence[preprocessed_sentence > 0]
    
    # Convert sentence back to words
    word_list = [id2word[word] for word in sentence]
    original_sentence = ' '.join(word_list)

    # Make the prediction
    prediction = model.predict(preprocessed_sentence)
    prediction = np.argmax(prediction[0], axis=1)

    # Map predicted tags to their respective entities
    prediction = prediction[:len(word_list)]
    pred_tag_list = [id2tag[tag_id] for tag_id in prediction]

    return original_sentence, pred_tag_list

# Load the pre-trained NER model
NER_model = tf.keras.models.load_model("Named_Entity_Recognition.keras")

# Title and text input for NER analysis
st.title("Named Entity Recognition (NER) Model")
st.markdown("Analyze sentences for named entities such as **Persons**, **Locations**, **Organizations**, and more!")

user_input = st.text_area("Enter a sentence for NER analysis:")

# Analyze the input sentence when the button is clicked
if st.button("Analyze"):
    if user_input.strip():
        preprocessed_sentence = preprocessingg([user_input])
        id2word_mapping = id2word()
        original_sentence, predicted_tags = make_prediction(NER_model, preprocessed_sentence, id2word_mapping, id2tag)

        # Color-coding for different named entities
        tag_colors = {
            'B-per': '#FF6347',  # Red for persons
            'I-per': '#FF6347',
            'B-geo': '#1E90FF',  # Blue for geographic entities
            'I-geo': '#1E90FF',
            'B-org': '#32CD32',  # Green for organizations
            'I-org': '#32CD32',
            'B-tim': '#FFD700',  # Yellow for time expressions
            'I-tim': '#FFD700',
            'O': '#808080'       # Gray for other words
        }

        # Display the result with inline color styling
        styled_sentence = []
        for word, tag in zip(original_sentence.split(), predicted_tags):
            color = tag_colors.get(tag, "#808080")
            styled_sentence.append(f"<span style='color:{color};'>{word}</span>")
        
        styled_sentence_html = ' '.join(styled_sentence)

        # Display the original sentence with color-coded entities
        st.markdown(f"### Original Sentence with Entities:")
        st.markdown(f"<p style='font-size:18px;'>{styled_sentence_html}</p>", unsafe_allow_html=True)
    else:
        st.write("Please enter a valid sentence.")
