# embedding_manager.py
import os
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config # Import settings from config.py

logger = logging.getLogger(__name__)

def _load_df(file_path: str, columns: list) -> pd.DataFrame:
    """Loads a dataframe from a pickle file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                return pd.DataFrame(pickle.load(f))
        except (pickle.UnpicklingError, EOFError, KeyError) as e:
            logger.error(f"Error loading or parsing {file_path}: {e}. Creating a new file.")
            return pd.DataFrame(columns=columns)
    return pd.DataFrame(columns=columns)

def _save_df(df: pd.DataFrame, file_path: str):
    """Saves a dataframe to a pickle file."""
    try:
        with open(file_path, "wb") as f:
            pickle.dump(df.to_dict(orient="list"), f)
    except Exception as e:
        logger.error(f"Error saving embeddings to {file_path}: {e}")
        raise

# --- User Management ---
def load_user_embeddings() -> pd.DataFrame:
    return _load_df(config.USER_EMBEDDINGS_FILE, ["user_id", "embedding"])

def save_user_embeddings(df: pd.DataFrame):
    _save_df(df, config.USER_EMBEDDINGS_FILE)

def find_duplicate_user(new_embedding: list):
    """Checks for a duplicate user face."""
    df = load_user_embeddings()
    if df.empty:
        return None

    all_embeddings = np.array(df["embedding"].tolist())
    similarities = cosine_similarity([new_embedding], all_embeddings)[0]
    
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    
    if best_score >= config.USER_SIMILARITY_THRESHOLD:
        return {"matched_user_id": df.iloc[best_idx]["user_id"], "score": float(best_score)}
    return None

# --- Employee Management ---
def load_employee_embeddings() -> pd.DataFrame:
    return _load_df(config.EMPLOYEE_EMBEDDINGS_FILE, ["employee_id", "employee_name", "embedding"])

def save_employee_embeddings(df: pd.DataFrame):
    _save_df(df, config.EMPLOYEE_EMBEDDINGS_FILE)

def check_is_employee(new_embedding: list):
    """Checks if a face belongs to a registered employee."""
    df = load_employee_embeddings()
    if df.empty:
        return None

    all_embeddings = np.array(df["embedding"].tolist())
    similarities = cosine_similarity([new_embedding], all_embeddings)[0]

    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score >= config.EMPLOYEE_SIMILARITY_THRESHOLD:
        return {
            "employee_id": df.iloc[best_idx]["employee_id"],
            "employee_name": df.iloc[best_idx]["employee_name"],
            "score": float(best_score)
        }
    return None
