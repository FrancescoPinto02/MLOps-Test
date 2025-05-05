import os
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from langdetect import detect, LangDetectException
from src.utils.logger import setup_logger  # Importa il logger

# Configura il logger
logger = setup_logger('ETL_Process', 'INFO')


# Funzione per caricare le variabili di ambiente
def load_environment():
    load_dotenv()
    logger.info("📂 Variabili d'ambiente caricate")


# Funzione per l'estrazione dei dati da MongoDB
def extract_data():
    logger.info("🔄 Estrazione dei dati da MongoDB...")

    MONGO_USER = os.getenv("MONGO_USER")
    MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
    MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")
    MONGO_DB = os.getenv("MONGO_DB")
    MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")

    mongo_uri = (f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_CLUSTER}"
                 f"/?retryWrites=true&w=majority&authSource={MONGO_AUTH_DB}")

    try:
        client = MongoClient(mongo_uri)
        db = client[MONGO_DB]
        games_cursor = db.games.find()
        reviews_cursor = db.reviews.find()
        games_df = pd.DataFrame(list(games_cursor))
        reviews_df = pd.DataFrame(list(reviews_cursor))
        logger.info("✅ Estrazione completata")
        return games_df, reviews_df
    except Exception as e:
        logger.error(f"❌ Errore nell'estrazione dei dati: {e}")
        raise


# Funzione per la trasformazione dei dati
def transform_data(games_df, reviews_df):
    logger.info("🔄 Trasformazione dei dati...")

    # 1. Rimuovere colonne non necessarie
    games_df.drop(columns=['_class', 'cover', 'screenshots', 'video'],
                  inplace=True, errors='ignore')

    reviews_df.drop(columns=['_class'], inplace=True, errors='ignore')

    # 2. Rinomina le colonne
    games_df.rename(columns={
        'title': 'game_title',
        'releaseDate': 'release_date',
        'rating': 'game_rating',
        'genre': 'game_genre',
        'platforms': 'game_platforms',
        'metaScore': 'game_meta_score',
        'metaScoreCount': 'game_meta_score_count'
    }, inplace=True)

    reviews_df.rename(columns={
        'author': 'review_author',
        'text': 'review_text',
        'score': 'review_score',
        'date': 'review_date',
        'gameId': 'game_id',
        'userId': 'user_id'
    }, inplace=True)

    # 3. Aggiungi la feature 'language' nelle recensioni
    def detect_language(text):
        try:
            return detect(text)
        except LangDetectException:
            return 'unknown'

    reviews_df['language'] = reviews_df['review_text'].apply(detect_language)

    logger.info("✅ Trasformazione completata")
    return games_df, reviews_df


# Funzione per caricare i dati trasformati (Load)
def load_data(games_df, reviews_df):
    logger.info("🔄 Caricamento dei dati trasformati...")

    # Salva i dati trasformati
    os.makedirs("data/processed", exist_ok=True)
    games_df.to_csv("data/processed/games_transformed.csv", index=False)
    reviews_df.to_csv("data/processed/reviews_transformed.csv", index=False)

    logger.info("✅ Caricamento completato: dati salvati in data/processed/")


# Funzione principale per orchestrare il processo ETL
def run_etl():
    logger.info("🚀 Inizio processo ETL...")
    load_environment()
    games_df, reviews_df = extract_data()
    games_df, reviews_df = transform_data(games_df, reviews_df)
    load_data(games_df, reviews_df)
    logger.info("✅ Processo ETL completato!")


# Esegui il processo ETL
if __name__ == "__main__":
    run_etl()
