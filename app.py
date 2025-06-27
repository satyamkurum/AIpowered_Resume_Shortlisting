import streamlit as st
st.set_page_config(page_title="Resume Fit Predictor", layout="centered") 

import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb
import torch
import joblib

# Load pre-trained components

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    return tokenizer, model

@st.cache_resource
def load_xgb_model():
    return joblib.load('xgb_model.pkl')

@st.cache_resource
def load_label_encoder():
    return joblib.load('label_encoder.pkl')

tokenizer, encoder_model = load_model_and_tokenizer()
clf = load_xgb_model()
le = load_label_encoder()

# Helper functions

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_embedding(text):
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        outputs = encoder_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

def predict_fit(resume_text, jd_text):
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(jd_text)
    res_embed = get_embedding(clean_resume)
    jd_embed = get_embedding(clean_jd)
    features = np.hstack((res_embed, jd_embed)).reshape(1, -1)
    pred = clf.predict(features)
    label = le.inverse_transform(pred)[0]
    return label

# Streamlit UI

st.title("Resume Fit Predictor")
st.write("Enter a resume and a job description to predict the candidate's fit level.")

resume_input = st.text_area("📄 Resume Text", height=200)
jd_input = st.text_area("📝 Job Description", height=200)

if st.button("Predict Fit"):
    if resume_input.strip() == "" or jd_input.strip() == "":
        st.warning("Please fill in both resume and job description.")
    else:
        with st.spinner("Analyzing..."):
            prediction = predict_fit(resume_input, jd_input)
        st.success(f"**Predicted Fit:** {prediction}")
