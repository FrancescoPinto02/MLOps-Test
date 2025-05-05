import pandas as pd
from langdetect import detect

def transform_data(games_df, reviews_df):
    # 1. Rimuovere colonne non necessarie nei due dataset
    games_df.drop(columns=['_class', 'cover', 'screenshots', 'video'], inplace=True, errors='ignore')
    reviews_df.drop(columns=['_class'], inplace=True, errors='ignore')

    # 2. Rinomina le colonne per avere nomi più convenzionali
    games_df.rename(columns={
        '_id': 'game_id',
        'title': 'game_title',
        'releaseDate': 'release_date',
        'rating': 'game_rating',
        'genre': 'game_genre',
        'platforms': 'game_platforms',
        'metaScore': 'game_meta_score',
        'metaScoreCount': 'game_meta_score_count'
    }, inplace=True)

    reviews_df.rename(columns={
        '_id': 'reviewer_id',
        'author': 'review_author',
        'text': 'review_text',
        'score': 'review_score',
        'date': 'review_date',
        'gameId': 'game_id',
        'userId': 'user_id'
    }, inplace=True)

    # 3. Aggiungi la feature 'language' nelle recensioni rilevando la lingua del testo
    def detect_language(text):
        try:
            return detect(text)
        except:
            return 'unknown'  # Se non si riesce a rilevare la lingua

    reviews_df['language'] = reviews_df['review_text'].apply(detect_language)

    # 4. Restituisci i dati trasformati
    return games_df, reviews_df

# Carica i dati raw
games_df = pd.read_csv("data/raw/games.csv")
reviews_df = pd.read_csv("data/raw/reviews.csv")

# Esegui le trasformazioni
games_transformed, reviews_transformed = transform_data(games_df, reviews_df)

# Salva i dati trasformati
games_transformed.to_csv("data/processed/games.csv", index=False)
reviews_transformed.to_csv("data/processed/reviews.csv", index=False)

print("Trasformazione completata: dati salvati in data/processed/games.csv e data/processed/reviews.csv")
