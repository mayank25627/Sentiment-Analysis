# Thing which have need to predict
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()


nltk.download('punkt')
nltk.download('stopwords')


# for transform the data

data = pd.read_csv('x_testings.csv', encoding='ISO-8859-1')

column_names = ['text']
data = pd.read_csv('x_testings.csv', names=column_names,
                   encoding=' ISO-8859-1')

X = data['text'].values
X_train = vectorizer.fit_transform(X)

# Here i load my Trained Model

loaded_model = pickle.load(open('trained_sentiment_analysis_model.sav', 'rb'))


# Preproceed Text Funtion

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


# generate predection funtion

def generatePredection(text):
    preprocessed_text = preprocess_text(text)
    sentences = preprocessed_text.split('\n')
    df = pd.DataFrame(sentences, columns=['Text'])
    new_pred = df['Text'].values
    X_pred_transform = vectorizer.transform(new_pred)

    return X_pred_transform


def predectionGenerationAndPrint(X_pred_transform):
    predictions = loaded_model.predict(X_pred_transform)

    for prediction in predictions:
        if prediction == 0:
            return ('Negative 😢')
        else:
            return ('Positive 😀')


# Write text on which you to predict output on.
text = "It is a very nice place"

preprocess_text_output = preprocess_text(text)
X_pred_transform = generatePredection(preprocess_text_output)
finalOutput = predectionGenerationAndPrint(X_pred_transform)


# Final Output result printed
print(finalOutput)


# Code for getting percentage
# positive_count = 0
# negative_count = 0

# Interpret the predictions and count occurrences

# predictions = loaded_model.predict(X_pred_transform)

# for prediction in predictions:
#     if prediction == 0:
#         negative_count += 1
#     else:
#         positive_count += 1


# print(positive_count, negative_count)

# # Calculate percentages
# total_predictions = len(predictions)
# positive_percentage = (positive_count / total_predictions) * 100
# negative_percentage = (negative_count / total_predictions) * 100

# print(f"Percentage of positive predictions: {positive_percentage}%")
# print(f"Percentage of negative predictions: {negative_percentage}%")
