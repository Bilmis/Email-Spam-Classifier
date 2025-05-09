# 📧 Email Spam Classifier

A machine learning application to classify emails as **Spam** or **Not Spam** using a deep learning model built with TensorFlow and Streamlit. This project is ideal for understanding text preprocessing, sequence modeling, and real-time ML app deployment.

---

## 🚀 Features

* Real-time spam detection
* Confidence score display
* Clean, modern Streamlit interface
* Preprocessing with stopwords removal, punctuation cleaning, and tokenization

---

## 🛠️ How It Works

1. **Text Cleaning:** Removes punctuation, stopwords, and lowercases the text for better model performance.
2. **Tokenization:** Converts text into numerical sequences.
3. **Padding:** Ensures uniform input length for the LSTM model.
4. **Prediction:** Uses a pre-trained LSTM model to classify the text as spam or not spam.
5. **Confidence Scoring:** Provides a confidence score for each prediction.

---

## 📦 Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/email-spam-classifier.git
cd email-spam-classifier
pip install -r requirements.txt
```

---

## 🚀 Run the App

Start the Streamlit server:

```bash
streamlit run streamlit_app.py
```

---

## 📂 Project Structure

```
📁 email-spam-classifier/
├── streamlit_app.py        # Main Streamlit app
├── spam_classifier_model.keras # Trained LSTM model
├── tokenizer.pkl           # Fitted tokenizer
├── requirements.txt        # Required libraries
└── README.md               # Project description (this file)
```

---

## ✨ Future Improvements

* Add visualizations like word clouds
* Include more advanced NLP techniques like BERT
* Deploy the app on a public server

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have ideas for improvement.

---