import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

vectorizer = TfidfVectorizer()


# for transform the data

data = pd.read_csv('x_testings.csv', encoding='ISO-8859-1')

column_names = ['text']
data = pd.read_csv('x_testings.csv', names=column_names,
                   encoding=' ISO-8859-1')

X = data['text'].values
X_train = vectorizer.fit_transform(X)
# Preprocess Text Function


def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


# Load Trained Model
loaded_model = pickle.load(open('trained_sentiment_analysis_model.sav', 'rb'))

# Generate Prediction Function


def generate_prediction(text, vectorizer):
    preprocessed_text = preprocess_text(text)
    X_pred_transform = vectorizer.transform([preprocessed_text])
    prediction = loaded_model.predict(X_pred_transform)
    confidence = loaded_model.predict_proba(
        X_pred_transform)[0]  # Extract confidence scores
    return prediction, confidence


# Write text on which you want to predict output
text = "It is a very nice place"

# Generate Prediction and Confidence
prediction, confidence = generate_prediction(text, vectorizer)

# Calculate Confidence Scores
positive_confidence = confidence[1] * 100  # Probability of positive class
negative_confidence = confidence[0] * 100  # Probability of negative class

# Print Output
print(f"Positive = {positive_confidence:.2f}% confidence")
print(f"Negative = {negative_confidence:.2f}% confidence")
