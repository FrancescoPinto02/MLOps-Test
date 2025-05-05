import os
import pandas as pd
import mlflow
import mlflow.sklearn
from flask.cli import load_dotenv
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from src.utils.logger import setup_logger
import dagshub
from scipy.stats import randint
import numpy as np

# Configura MLFlow Tracking Server
load_dotenv()
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(mlflow_uri)

# Configura il logger
logger = setup_logger('Collaborative Filtering', 'INFO')


# Funzione di data cleaning e data preparation
def clean_and_prepare_data(reviews_df):
    logger.info("Inizio data cleaning e preparation...")

    # Rimuovere recensioni con punteggio mancante o negativo
    reviews_df = reviews_df.dropna(subset=['review_score'])
    reviews_df = reviews_df[reviews_df['review_score'] >= 0]

    # Aggregare i punteggi per ogni coppia user_id, game_id
    reviews_df = reviews_df.groupby(['user_id', 'game_id'], as_index=False)['review_score'].mean()

    # Creare la matrice di valutazione (user-item matrix)
    rating_matrix = reviews_df.pivot(index='user_id', columns='game_id', values='review_score')

    logger.info(
        f"Matrice di valutazione preparata con {rating_matrix.shape[0]} utenti e {rating_matrix.shape[1]} giochi.")

    return rating_matrix


# Funzione di addestramento del modello Collaborative Filtering (SVD)
def train_collaborative_filtering(rating_matrix, n_components=20, n_iter=5):
    logger.info("Inizio addestramento del modello di Collaborative Filtering...")

    # Impostare SVD (Singular Value Decomposition)
    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=42)

    # Adattare il modello sulla matrice di valutazione
    svd.fit(rating_matrix.fillna(0))

    logger.info(f"Modello addestrato con {n_components} componenti e {n_iter} iterazioni.")

    return svd


# Funzione di valutazione del modello (RMSE)
def evaluate_model(svd, rating_matrix):
    logger.info("Valutazione del modello...")

    # Predire le valutazioni
    predicted_ratings = svd.transform(rating_matrix.fillna(0)).dot(svd.components_)

    # Calcolare RMSE
    mse = mean_squared_error(rating_matrix.fillna(0), predicted_ratings)
    rmse = mse ** 0.5

    logger.info(f"RMSE del modello: {rmse}")

    return rmse


# Funzione di scoring personalizzata per calcolare l'RMSE
def rmse_scorer(estimator, X, y):
    # Estimare le valutazioni con il modello
    predicted_ratings = estimator.transform(X.fillna(0)).dot(estimator.components_)

    # Calcolare RMSE
    mse = mean_squared_error(y.fillna(0), predicted_ratings)
    rmse = np.sqrt(mse)
    return rmse


# Funzione per la ricerca degli iperparametri con GridSearchCV
def grid_search(rating_matrix):
    logger.info("Inizio GridSearchCV per l'ottimizzazione degli iperparametri...")

    # Impostare la griglia di iperparametri
    param_grid = {
        'n_components': [10, 20, 30],  # Cambiato da 'n_latent_factors' a 'n_components'
        'n_iter': [5, 10, 15]
    }

    # Creare il modello SVD
    svd = TruncatedSVD(random_state=42)

    # Creare il GridSearchCV
    grid_search = GridSearchCV(estimator=svd, param_grid=param_grid, cv=3, scoring=rmse_scorer)

    # Esegui la ricerca
    grid_search.fit(rating_matrix.fillna(0), rating_matrix)  # Passiamo anche y_train (rating_matrix)

    logger.info(f"Migliori parametri trovati: {grid_search.best_params_}")
    logger.info(f"RMSE migliore: {grid_search.best_score_}")

    return grid_search.best_params_, grid_search.best_score_


# Funzione per la ricerca degli iperparametri con RandomizedSearchCV
def random_search(rating_matrix):
    logger.info("Inizio RandomizedSearchCV per l'ottimizzazione degli iperparametri...")

    # Impostare la distribuzione degli iperparametri
    param_dist = {
        'n_components': randint(5, 50),  # Cambiato da 'n_latent_factors' a 'n_components'
        'n_iter': randint(5, 20)
    }

    # Creare il modello SVD
    svd = TruncatedSVD(random_state=42)

    # Creare il RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=svd, param_distributions=param_dist, n_iter=10, cv=3,
                                       scoring=rmse_scorer, random_state=42)

    # Esegui la ricerca
    random_search.fit(rating_matrix.fillna(0), rating_matrix)  # Passiamo anche y_train (rating_matrix)

    logger.info(f"Migliori parametri trovati: {random_search.best_params_}")
    logger.info(f"RMSE migliore: {random_search.best_score_}")

    return random_search.best_params_, random_search.best_score_


# Funzione principale per eseguire il processo
def run_collaborative_filtering_pipeline(reviews_df, n_components=20, n_iter=5, search_method='grid'):
    # Pulire e preparare i dati
    rating_matrix = clean_and_prepare_data(reviews_df)

    # Dividere il dataset in train e test (80% - 20%)
    train_matrix, test_matrix = train_test_split(rating_matrix, test_size=0.2, random_state=42)

    # Iniziare il tracciamento con MLflow
    with mlflow.start_run():
        # Disabilita autologging
        mlflow.sklearn.autolog(disable=True)

        # Esegui la ricerca degli iperparametri
        if search_method == 'grid':
            best_params, best_rmse = grid_search(train_matrix)
        elif search_method == 'random':
            best_params, best_rmse = random_search(train_matrix)

        # Log dei migliori parametri e del RMSE
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", best_rmse)

        logger.info(f"Esperimento completato con RMSE: {best_rmse} e parametri: {best_params}")


if __name__ == "__main__":
    # Carica il dataset delle recensioni
    reviews_df = pd.read_csv("../../data/processed/reviews.csv")

    # Esegui la pipeline di Collaborative Filtering con GridSearchCV (modifica 'grid' a 'random' per usare RandomizedSearchCV)
    run_collaborative_filtering_pipeline(reviews_df, n_components=20, n_iter=10, search_method='grid')
