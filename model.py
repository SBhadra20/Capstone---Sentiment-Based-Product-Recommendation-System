#Capstone Project: Ebuss - Sentiment-Based Product Recommendation System
#Name: Suraj Bhadra, Cohort: C 75

import pandas as pd
import numpy as np
import re
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

_nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
os.makedirs(_nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, _nltk_data_dir)
nltk.download('stopwords', download_dir=_nltk_data_dir, quiet=True)
nltk.download('wordnet', download_dir=_nltk_data_dir, quiet=True)


# Text Preprocessing

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)



# Load or Train Models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pkl_files = [
    'sentiment_model.pkl', 'tfidf_vectorizer.pkl',
    'user_item_matrix.pkl', 'predicted_ratings.pkl', 'cleaned_data.pkl'
]
pickles_exist = all(os.path.exists(os.path.join(BASE_DIR, f)) for f in pkl_files)

if pickles_exist:
    print("Loading pre-trained models from pickle files...")
    sentiment_model = pickle.load(open(os.path.join(BASE_DIR, 'sentiment_model.pkl'), 'rb'))
    tfidf_vectorizer = pickle.load(open(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'), 'rb'))
    user_item_matrix = pickle.load(open(os.path.join(BASE_DIR, 'user_item_matrix.pkl'), 'rb'))
    predicted_ratings = pickle.load(open(os.path.join(BASE_DIR, 'predicted_ratings.pkl'), 'rb'))
    df = pd.read_pickle(os.path.join(BASE_DIR, 'cleaned_data.pkl'))
    print("Models loaded successfully.")
else:
    print("Pickle files not found. Training models from scratch...")

    # Load and clean data
    df = pd.read_csv(os.path.join(BASE_DIR, 'sample30.csv'))
    df.drop(columns=['id', 'reviews_userCity', 'reviews_userProvince',
                      'reviews_didPurchase', 'manufacturer'], inplace=True, errors='ignore')
    df.dropna(subset=['reviews_username', 'reviews_text', 'user_sentiment'], inplace=True)
    df['reviews_doRecommend'] = df['reviews_doRecommend'].fillna(True)
    df['reviews_rating'] = df['reviews_rating'].astype(int)
    df.drop_duplicates(inplace=True)
    df['reviews_title'] = df['reviews_title'].fillna('')
    df['combined_review'] = df['reviews_title'].astype(str) + ' ' + df['reviews_text'].astype(str)
    df['processed_review'] = df['combined_review'].apply(preprocess_text)
    df['sentiment'] = df['user_sentiment'].map({'Positive': 1, 'Negative': 0})

    # Sentiment Model (Logistic Regression + TF-IDF) 
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = tfidf_vectorizer.fit_transform(df['processed_review'])
    y = df['sentiment']
    sentiment_model = LogisticRegression(C=1, penalty='l2', solver='liblinear',
                                         max_iter=1000, random_state=42)
    sentiment_model.fit(X_tfidf, y)

    # User-Based Recommendation System 
    user_item_matrix = df.pivot_table(
        index='reviews_username', columns='name',
        values='reviews_rating', aggfunc='mean'
    )
    filled = user_item_matrix.fillna(0)
    user_sim = cosine_similarity(filled)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index,
                               columns=user_item_matrix.index)

    R = filled.values
    M = user_item_matrix.notna().astype(float).values
    S = user_sim_df.values.copy()
    np.fill_diagonal(S, 0)

    num = S @ R
    den = np.abs(S) @ M
    den[den == 0] = 1
    pred = num / den

    predicted_ratings = pd.DataFrame(pred, index=user_item_matrix.index,
                                     columns=user_item_matrix.columns)

    # Save for next startup
    pickle.dump(sentiment_model, open(os.path.join(BASE_DIR, 'sentiment_model.pkl'), 'wb'))
    pickle.dump(tfidf_vectorizer, open(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'), 'wb'))
    pickle.dump(user_item_matrix, open(os.path.join(BASE_DIR, 'user_item_matrix.pkl'), 'wb'))
    pickle.dump(predicted_ratings, open(os.path.join(BASE_DIR, 'predicted_ratings.pkl'), 'wb'))
    df.to_pickle(os.path.join(BASE_DIR, 'cleaned_data.pkl'))
    print("Models trained and saved.")



# Recommendation Functions

valid_usernames = sorted(user_item_matrix.index.tolist())


def get_top20_recommendations(username):
    if username not in user_item_matrix.index:
        return []

    user_ratings = user_item_matrix.loc[username]
    unrated = user_ratings[user_ratings.isna()].index.tolist()

    if len(unrated) == 0:
        return predicted_ratings.loc[username].nlargest(20).index.tolist()

    preds = predicted_ratings.loc[username, unrated].sort_values(ascending=False)
    return preds.head(20).index.tolist()


def filter_top5_by_sentiment(products):
    scores = {}
    for product in products:
        reviews = df[df['name'] == product]['processed_review']
        if len(reviews) == 0:
            scores[product] = 0.5
            continue
        X = tfidf_vectorizer.transform(reviews)
        preds = sentiment_model.predict(X)
        scores[product] = preds.mean()

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [p for p, _ in ranked[:5]]


def recommend_products(username):
    top20 = get_top20_recommendations(username)
    if not top20:
        return []
    return filter_top5_by_sentiment(top20)
