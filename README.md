#  Fakebuster – A Deep Learning-Based Fake News Detection System

Welcome to **Fakebuster**, an AI-powered system that classifies news headlines and articles as **True** or **Fake**.  
This project uses **Python**, **TensorFlow**, and **pre-trained GloVe embeddings** to build a **hybrid LSTM-CNN model** for fake news detection.

This project demonstrates:
- Basics of **Natural Language Processing (NLP)**
- Usage of **pre-trained word embeddings (GloVe)**
- Building and training a **deep learning text classifier**
- End-to-end **data preprocessing, model training, and prediction**

---

## Features

- Preprocessing of textual data (titles and articles)
- Label encoding for categorical classification
- Uses **pre-trained GloVe embeddings** for semantic understanding
- Hybrid **CNN-LSTM architecture** for robust sequence modeling
- Predictive analysis for unseen news headlines
- Example interface for quick single-news prediction

---

## Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy & Pandas**
- **Scikit-learn**
- **GloVe pre-trained embeddings**

---

## How It Works

Fake-News-Detector operates in the following steps:

1. **Data Preprocessing**  
   - Tokenizes and pads news titles/articles
   - Encodes labels as numerical values

2. **Embedding Layer**  
   - Uses pre-trained **GloVe embeddings** to convert words into dense vectors

3. **Hybrid CNN-LSTM Model**  
   - **Conv1D + MaxPooling**: Extracts local patterns from sequences
   - **LSTM**: Captures long-term dependencies
   - **Dense layer**: Sigmoid output for binary classification

4. **Prediction**  
   - Input a news headline/text
   - Model predicts **True** or **Fake**

Example of usage:

```python
X = "The government announced new tax reforms today"
sequences = tokenizer1.texts_to_sequences([X])
sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
prediction = model.predict(sequences, verbose=0)[0][0]

if prediction >= 0.5:
    print("This news is True")
else:
    print("This news is False")
