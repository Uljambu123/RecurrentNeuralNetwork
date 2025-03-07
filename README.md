# RecurrentNeuralNetwork
**Project Overview**
This project is a Recurrent Neural Network (RNN) model that performs sentiment analysis on IMDB movie reviews. It uses a Simple RNN architecture to classify reviews as either positive or negative based on their textual content.

**Objective**
The objective of this project is to develop a deep learning model using RNN to analyze movie reviews from the IMDB dataset and classify them into positive or negative sentiments. The project also includes a Streamlit-based web application for user interaction.

**Key Steps in the Project**
**1. _Dataset Preparation_**
a. The IMDB dataset is loaded using TensorFlow's imdb module.
b. The dataset is tokenized, and a word index is created.
c. Each review is encoded as a sequence of integers.

**2. _Preprocessing_**
a. Custom preprocessing functions convert raw text into tokenized sequences.
b. Reviews are padded to a fixed length (500 words) to ensure uniform input dimensions.

**3. _Model Loading_**
a. A pre-trained Simple RNN model (simple_rnn_imdb.h5) is loaded.
b. The model uses ReLU activation and outputs a probability score for sentiment classification.

**4. _Streamlit Web App Development_**

The web application allows users to input movie reviews.
The review is processed, and the model predicts the sentiment.
The predicted sentiment and confidence score are displayed.

_**Conclusions**_
a. RNNs can effectively capture sequential dependencies in text and are suitable for sentiment analysis tasks.
b. Streamlit provides a user-friendly interface to deploy deep learning models for real-time prediction.
c. The model successfully predicts positive or negative sentiment based on the probability threshold (>0.5 for positive and <=0.5 for negative).
d. The use of pre-trained models improves efficiency, reducing the need for retraining.

**_Technologies Used_**
a. Python for scripting
b. TensorFlow & Keras for deep learning
c. Streamlit for the web-based user interface
d. NumPy & Pandas for data handling
e. Scikit-learn for additional ML utilities
f. Matplotlib for visualization (if needed)

**How to Run**
Prerequisites
Install dependencies using:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run main.py

Open the Streamlit UI, enter a movie review, and get sentiment classification.

**Future Work**
a. Improve model accuracy by using LSTM or GRU instead of Simple RNN.
b. Fine-tune embeddings for better word representation.
c. Expand dataset with more real-world movie reviews.
d. Deploy the app on a cloud service like AWS or Heroku.
e. Implement a confidence interval for sentiment prediction.

