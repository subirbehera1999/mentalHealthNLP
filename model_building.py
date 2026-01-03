# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from utils.text_cleaner import TextCleanerTFIDF
import joblib

# import dataset 
df = pd.read_csv("data/mental_health_imbalanced.csv")

# remove redundant feature and duplicate record
df = df.drop("Unique_ID",axis=1)
if df.duplicated().sum() > 0:
    df = df.drop_duplicates()

# Separate X and y variable
x = df.drop(["status"],axis=1)
y = df["status"]

# Build pipeline with TF-IDF Vectorizer 
pipeline = Pipeline(
    steps=[
        ("cleaner", TextCleanerTFIDF()),
        ("vectorizer", TfidfVectorizer(ngram_range=(1, 1),min_df=5,max_df=0.9,max_features=20000,sublinear_tf=True)),
        ("model", LogisticRegression(max_iter=2000,C=1.0,solver="liblinear",class_weight="balanced"
        ))
    ]
)

# fit model (it is final model so no need to train and test)
pipeline.fit(x,y)

# save the model 
joblib.dump(pipeline,"models/model.pkl")