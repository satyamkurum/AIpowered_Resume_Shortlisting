# AI-Powered Resume Shortlisting System

Hello Folks, This is an intelligent machine learning-based system that evaluates resumes based on Resume Data and Job Description and classifies candidates into **Good Fit**, **Potential Fit**, or **No Fit** based on training data. This project leverages XGBoost for classification, includes model serialization, and exposes functionality via a web application.

## Table of Contents

- Overview
- Demo
- Project Structure
- How It Works
- Setup & Installation
- Model Performance
- Setup and Installation

## Overview

The Resume Shortlisting System automates candidate evaluation using Natural Language Processing and Machine Learning. It accepts Resume Data and Job Description, processes it through a trained XGBoost model, and predicts the category of fit for a given job role. The project is built to be modular, extensible, and easy to deploy.


## Demo

Coming Soon...

## Project Structure

`Resume_Analyzer.ipynb` - Jupyter Notebook for training, evaluating, and saving the ML model.
`app.py`                - Backend application file(Streamlit) that serves the ML model and provides an UI.
`train.csv`             - Labeled training dataset used to train the model. 
`test.csv`              - Test dataset used to evaluate model performance. |
`xgb_model.pkl`         - Serialized XGBoost model saved using `joblib`. Used during deployment.
`label_encoder.pkl`     - Encoded class labels (e.g., Good Fit, No Fit) saved for consistent inference.
`Web UI.pdf`            - visual mockup of the user interface design. 
`requirements.txt`      - Python package dependencies required to run this project.


## How It Works

   **Data Preprocessing**
    - Raw resume data is cleaned and encoded using a `LabelEncoder`.

   **Model Training**
    - XGBoost classifier is trained on labeled resume features from `train.csv`.

   **Evaluation**
    - Predictions are generated and evaluated using metrics such as Precision, Recall, and F1-score on `test.csv`.

   **Serialization**
     - Trained model (`xgb_model.pkl`) and label encoder (`label_encoder.pkl`) are stored using `joblib`.

   **Deployment**
     - `app.py` loads the model and exposes a simple interface or API for real-time predictions.

---

## Model Performance

**Classification Report (Sample)**

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Good Fit      | 0.65      | 0.76   | 0.70     | 303     |
| No Fit        | 0.73      | 0.77   | 0.75     | 658     |
| Potential Fit | 0.68      | 0.49   | 0.57     | 288     |
| **Accuracy**  |           |        | **0.70** | 1249    |


## Setup & Installation

### Clone the Repository

```bash
git https://github.com/satyamkurum/AIpowered_Resume_Shortlisting.git

### Install Dependencies
pip install -r requirements.txt
```
## About Me
  Satyam Kurum
- Data Scientist | ML Developer | 2025 NITK Surathkal Graduate
- Passionate about GenAI, NLP, and creative machine learning apps

- You are free to use, modify, and distribute it with attribution.

### Run the App
python app.py
Or Using
streamlit run app.py
