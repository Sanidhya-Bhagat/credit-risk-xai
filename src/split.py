from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE
from src.data import load_dataset


def create_train_test_split():
    """Create a reproducible stratified train/test split."""

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test