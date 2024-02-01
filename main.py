from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


app = FastAPI()


# Load the trained model
with open("D:/code/stacking_classifier_model.joblib", "rb") as file:
    model = joblib.load(file)

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('D:\code\tfidf_vectorizer.joblib')


class Item(BaseModel):
    text: str


@app.post("/model")
async def predict(item: Item):
    # Preprocess the input text using the loaded TF-IDF vectorizer
    text_tfidf = tfidf_vectorizer.transform([item.text])

    # Make predictions using the stacked model
    prediction = model.predict(text_tfidf)

    return {"prediction": int(prediction[0])}

