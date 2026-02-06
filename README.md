🎭 Emotion Detection from Text
Streamlit + TF-IDF + XGBoost

A fast and lightweight Machine Learning web app that detects human emotions from text using TF-IDF vectorization and an XGBoost classifier, wrapped inside an interactive Streamlit UI.

Users simply enter a sentence and instantly get the predicted emotion.

🚀 Features

✅ Real-time emotion prediction
✅ Clean and modern Streamlit interface
✅ Custom CSS styling
✅ Fast inference (XGBoost)
✅ Lightweight and runs locally
✅ Command-line testing supported
✅ Easy to deploy

🧠 Supported Emotions

The model predicts:

Joy

Sadness

Anger

Fear

Love

Surprise

⚙️ Tech Stack
Component	Technology
Language	Python
ML Model	XGBoost
Text Features	TF-IDF
Frontend	Streamlit
Model Saving	Joblib
Encoding	LabelEncoder
📂 Project Structure
emotion-detection-app/
│
├── app.py
├── Xg_Boost_model.pkl
├── vectorizer.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md

🔄 Workflow
User Text
   ↓
TF-IDF Vectorizer
   ↓
XGBoost Model
   ↓
Label Decoder
   ↓
Emotion Prediction

🛠️ Installation
Step 1 — Clone repo
git clone https://github.com/yourusername/emotion-detection-app.git
cd emotion-detection-app

Step 2 — Install dependencies
pip install -r requirements.txt


If no requirements file:

pip install streamlit scikit-learn xgboost joblib

▶️ Run the App
streamlit run app.py


Then open:

http://localhost:8501

💻 Usage
Web Interface

Enter text

Click Analyze Emotion

View predicted emotion

Example
Input: I am very happy today!
Output: Joy

🧪 Command Line Testing (Optional)
python app.py

Enter Text Here: I feel nervous
Predicted Emotion: Fear

🎨 UI Highlights

Soft background theme

Styled text area

Hover buttons

Centered layout

Large highlighted emotion result

📈 Possible Improvements

You can extend this project with:

LSTM / BERT models

Voice emotion detection

Multilingual support

REST API (FastAPI/Flask)

Mobile app

Emotion analytics dashboard

Docker deployment

📊 Use Cases

Sentiment analysis

Mental health monitoring

Chatbots

Social media analytics

Customer feedback classification

NLP learning projects

👨‍💻 Author

Iftikhar Ul Hassan
