# Recurrent Neural Network (RNN) - IMDB Sentiment Analysis

## 📌 Project Overview
This project is a **Recurrent Neural Network (RNN) model** that performs **sentiment analysis on IMDB movie reviews**. It utilizes a **Simple RNN architecture** to classify reviews as either **positive or negative** based on their textual content. Additionally, a **Streamlit-based web application** allows users to input movie reviews and receive sentiment predictions.

---

## 🎯 Objective
The objective of this project is to develop a **deep learning model** using **RNN** to analyze movie reviews from the **IMDB dataset** and classify them into **positive or negative sentiments**. The project also includes a **Streamlit-based web application** for user interaction.

---

## 🔑 Key Steps in the Project
### 1️⃣ Dataset Preparation
- The **IMDB dataset** is loaded using TensorFlow's `imdb` module.
- The dataset is tokenized, and a **word index** is created.
- Each review is encoded as a sequence of integers.

### 2️⃣ Preprocessing
- Custom **preprocessing functions** convert raw text into tokenized sequences.
- Reviews are **padded to a fixed length** (500 words) to ensure uniform input dimensions.

### 3️⃣ Model Loading
- A **pre-trained Simple RNN model** (`simple_rnn_imdb.h5`) is loaded.
- The model uses **ReLU activation** and outputs a probability score for sentiment classification.

### 4️⃣ Streamlit Web App Development
- The web application allows users to **input movie reviews**.
- The review is processed, and the **model predicts the sentiment**.
- The predicted sentiment and **confidence score** are displayed.

---

## 📌 Conclusions
- **RNNs can effectively capture sequential dependencies in text** and are suitable for **sentiment analysis** tasks.
- **Streamlit provides a user-friendly interface** to deploy deep learning models for real-time prediction.
- The model successfully predicts **positive or negative sentiment** based on the probability threshold (**>0.5 for positive** and **≤0.5 for negative**).
- The use of **pre-trained models improves efficiency**, reducing the need for retraining.

---

## 🛠️ Technologies Used
- **Python** for scripting
- **TensorFlow & Keras** for deep learning
- **Streamlit** for the web-based user interface
- **NumPy & Pandas** for data handling
- **Scikit-learn** for additional ML utilities
- **Matplotlib** for visualization (if needed)

---

## 🚀 How to Run

### 📌 Prerequisites
1. **Clone this repository:**
   ```bash
   git clone https://github.com/your-repository-name.git
   cd your-repository-name

2. **Install dependencies using:**
   ```bash
   pip install -r requirements.txt

3. **Run the Streamlit app:**
   ```bash
   streamlit run main.py

4. **Run the Streamlit app:**
   ```bash
   Open the Streamlit UI, enter a movie review, and get sentiment classification.


## 🛠️ Future Work
- Improve model accuracy by using LSTM or GRU instead of Simple RNN.
- Fine-tune embeddings for better word representation.
- Expand dataset with more real-world movie reviews.
- Deploy the app on a cloud service like AWS or Heroku.
- Implement a confidence interval for sentiment prediction.
