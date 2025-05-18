import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

# Initialize NLTK resources
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)

# Initialize session state for models and vectorizer
if 'lr_model' not in st.session_state:
    st.session_state.lr_model = None
if 'nn_model' not in st.session_state:
    st.session_state.nn_model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'accuracy_lr' not in st.session_state:
    st.session_state.accuracy_lr = None
if 'accuracy_nn' not in st.session_state:
    st.session_state.accuracy_nn = None

# Simple text preprocessing
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return ' '.join(tokens)

# Streamlit app
st.title("Fake News Detection System")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Train Model", "Predict"])

if page == "Train Model":
    st.header("Train Model")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")
    
    if uploaded_file is not None:
        # Read dataset
        start = time.time()
        df = pd.read_csv(uploaded_file)
        st.write(f"Dataset loaded in {time.time() - start:.2f} seconds")
        
        # Input column names
        text_col = st.text_input("Enter the name of the text column", value="text")
        label_col = st.text_input("Enter the name of the label column", value="label")
        
        if text_col in df.columns and label_col in df.columns:
            # Preprocess data
            start = time.time()
            df['processed_text'] = df[text_col].apply(preprocess_text)
            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(df['processed_text'])
            y = df[label_col].map({'FAKE': 0, 'REAL': 1})
            st.write(f"Preprocessing done in {time.time() - start:.2f} seconds")
            
            # Train models
            if st.button("Train Models"):
                # Split data (simple split for small dataset)
                train_size = int(0.8 * X.shape[0])
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                # Logistic Regression
                start = time.time()
                lr_model = LogisticRegression(max_iter=100)
                lr_model.fit(X_train, y_train)
                st.session_state.lr_model = lr_model
                lr_pred = lr_model.predict(X_test)
                st.session_state.accuracy_lr = accuracy_score(y_test, lr_pred)
                st.write(f"Logistic Regression trained in {time.time() - start:.2f} seconds")
                st.write(f"Logistic Regression Accuracy: {st.session_state.accuracy_lr:.2f}")
                
                # Neural Network
                start = time.time()
                nn_model = Sequential([
                    Dense(16, activation='relu', input_shape=(X.shape[1],)),
                    Dense(1, activation='sigmoid')
                ])
                nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                nn_model.fit(X_train.toarray(), y_train, epochs=5, batch_size=32, verbose=0)
                st.session_state.nn_model = nn_model
                nn_pred = (nn_model.predict(X_test.toarray(), verbose=0) > 0.5).astype(int).flatten()
                st.session_state.accuracy_nn = accuracy_score(y_test, nn_pred)
                st.write(f"Neural Network trained in {time.time() - start:.2f} seconds")
                st.write(f"Neural Network Accuracy: {st.session_state.accuracy_nn:.2f}")
                
                # Store vectorizer for predictions
                st.session_state.vectorizer = vectorizer
                
                # Confusion Matrix Visualization
                st.subheader("Confusion Matrix")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                # Logistic Regression Confusion Matrix
                cm_lr = confusion_matrix(y_test, lr_pred)
                sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax1)
                ax1.set_title("Logistic Regression")
                ax1.set_xlabel("Predicted")
                ax1.set_ylabel("Actual")
                
                # Neural Network Confusion Matrix
                cm_nn = confusion_matrix(y_test, nn_pred)
                sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues', ax=ax2)
                ax2.set_title("Neural Network")
                ax2.set_xlabel("Predicted")
                ax2.set_ylabel("Actual")
                
                st.pyplot(fig)
                
                st.success("Models trained successfully! Switch to the 'Predict' page to make predictions.")
        else:
            st.error("Specified columns not found in the dataset.")
else:
    st.header("Predict")
    
    if st.session_state.lr_model is None or st.session_state.nn_model is None or st.session_state.vectorizer is None:
        st.warning("Please train the models first on the 'Train Model' page.")
    else:
        # Input for prediction
        user_input = st.text_area("Enter text to predict if it's fake or real news", height=200)
        
        if st.button("Predict"):
            if user_input.strip() == "":
                st.error("Please enter some text to predict.")
            else:
                # Preprocess input
                start = time.time()
                processed_input = preprocess_text(user_input)
                X_input = st.session_state.vectorizer.transform([processed_input])
                
                # Logistic Regression prediction
                lr_pred = st.session_state.lr_model.predict(X_input)[0]
                lr_label = 'REAL' if lr_pred == 1 else 'FAKE'
                
                # Neural Network prediction
                nn_pred = (st.session_state.nn_model.predict(X_input.toarray(), verbose=0) > 0.5).astype(int)[0][0]
                nn_label = 'REAL' if nn_pred == 1 else 'FAKE'
                
                # Display results
                st.write(f"**Logistic Regression Prediction**: {lr_label}")
                st.write(f"**Neural Network Prediction**: {nn_label}")
                st.write(f"Prediction time: {time.time() - start:.2f} seconds")
                
                # Display training accuracy
                st.subheader("Model Performance (from Training)")
                st.write(f"Logistic Regression Accuracy: {st.session_state.accuracy_lr:.2f}")
                st.write(f"Neural Network Accuracy: {st.session_state.accuracy_nn:.2f}")
