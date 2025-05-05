import os
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd

# Carica le variabili d'ambiente
load_dotenv()

MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")

# URI di connessione a MongoDB Atlas
mongo_uri = f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_CLUSTER}/?retryWrites=true&w=majority&authSource={MONGO_AUTH_DB}"

# Connetti al client
client = MongoClient(mongo_uri)
db = client[MONGO_DB]

# Estrai le collection
games_cursor = db.games.find()
reviews_cursor = db.reviews.find()

# Converti in DataFrame
games_df = pd.DataFrame(list(games_cursor))
reviews_df = pd.DataFrame(list(reviews_cursor))

# Salva in CSV in data/raw/
os.makedirs("data/raw", exist_ok=True)
games_df.to_csv("data/raw/games.csv", index=False)
reviews_df.to_csv("data/raw/reviews.csv", index=False)

print("✅ Estrazione completata: dati salvati in data/raw/")