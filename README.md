# Ebuss — Sentiment-Based Product Recommender
 
An end-to-end ML system that combines collaborative filtering with NLP-driven sentiment analysis to surface the 5 best-fit products for any registered user — deployed via Flask.
 
![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-green) ![XGBoost](https://img.shields.io/badge/XGBoost-green) ![TF--IDF](https://img.shields.io/badge/TF--IDF-orange) ![Collaborative Filtering](https://img.shields.io/badge/Collaborative_Filtering-orange)
 
---
 
## Dataset
 
| Metric | Value |
|---|---|
| Product reviews | 30,000 |
| Unique products | 200+ |
| Users | 20,000+ |
| Feature columns | 15 |
 
Source: `sample30.csv` — a subset of the [Kaggle e-commerce reviews dataset](https://www.kaggle.com/).
 
---
 
## Overview
 
Ebuss competes in the e-commerce space against platforms like Amazon and Flipkart. This project builds a sentiment-aware recommendation engine — it first surfaces 20 candidate products per user using collaborative filtering on historical ratings, then re-ranks them by the proportion of positive sentiments across each product's reviews, returning the top 5. The final model is served through a minimal Flask UI.
 
---
 
## Pipeline
 
| Step | Description |
|---|---|
| 1. Data cleaning & EDA | Null handling, deduplication, dtype normalisation, class balance check on the `user_sentiment` target |
| 2. Text preprocessing | Merge `reviews_text` + `reviews_title`, lowercase, strip punctuation, remove stopwords, lemmatize |
| 3. Feature extraction | Stratified train/test split; TF-IDF vectorisation with saved vectorizer pickle for deployment |
| 4. Sentiment model | Three classifiers evaluated (Logistic Regression, Random Forest, XGBoost / Naive Bayes); best model selected on F1 with SMOTE if imbalanced |
| 5. Recommendation system | User-based and item-based collaborative filtering built on the `reviews_username × name` rating pivot; evaluated via RMSE; best system selected |
| 6. Top-20 candidates | For a given username, retrieve 20 product recommendations from the selected recommendation engine |
| 7. Sentiment re-ranking | Run all reviews of the 20 candidates through the sentiment model; rank by % positive reviews; return top 5 |
| 8. Flask deployment | Username input → submit → 5 product recommendations rendered in the browser |
 
---
 
## Repository Structure
 
```
├── notebook.ipynb           # End-to-end analysis — EDA through recommendation fine-tuning
├── model.py                 # Loads pickled model + recommendation system; exposes get_recommendations()
├── app.py                   # Flask app — routes / (GET form) and / (POST results)
├── tfidf.pkl                # Fitted TF-IDF vectorizer
├── sentiment_model.pkl      # Best trained sentiment classifier
├── user_item_matrix.pkl     # Rating pivot matrix for collaborative filtering
├── item_similarity.pkl      # Pre-computed item-item cosine similarity matrix
├── sample30.csv             # Source dataset (30k reviews, 15 columns)
└── requirements.txt         # Python dependencies
```
 
---
 
## Quickstart
 
```bash
pip install -r requirements.txt
python app.py
# Open http://127.0.0.1:5000 → enter a username → get 5 recommendations
```
 
---
 
## Key Dependencies
 
```
flask
pandas
numpy
scikit-learn
xgboost
nltk
imbalanced-learn
scipy
pickle
```
 
---
 
## How It Works
 
### Sentiment Analysis
Reviews are preprocessed (lowercased, stopwords removed, lemmatized) and converted to TF-IDF feature vectors. A classification model predicts whether each review sentiment is **Positive** or **Negative**.
 
### Recommendation System
A user-item rating matrix is constructed from `reviews_username` and product `name` columns with `reviews_rating` as values. Cosine similarity is computed between items (or users), and the top-20 unrated products are recommended for a given user.
 
### Fine-Tuning with Sentiment
For each of the 20 recommended products, all available reviews are passed through the sentiment model. Products are ranked by their **% positive review score**, and the top 5 are returned to the user.
 
---
 
## Assumptions
 
- The model is built solely on users and products already present in the dataset.
- No new users or products are introduced at inference time.
- Recommendations are only generated for usernames that exist in `reviews_username`.
---
 
## Evaluation Rubric Summary
 
| Task | Weight |
|---|---|
| Data cleaning & preprocessing | 10% |
| Text preprocessing | 10% |
| Feature extraction | 10% |
| Model building (≥3 classifiers) | 20% |
| Recommendation system (user-based + item-based) | 20% |
| Top-20 product recommendations | 10% |
| Sentiment-based fine-tuning (top 5) | 10% |
| Flask deployment | 10% |
 
---
 
## Author
 
Built as a capstone project for the PG Diploma in AI & ML — IIIT Bangalore.
