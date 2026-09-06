from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE
from src.split import create_train_test_split


def create_train_validation_test_split():
    """Create reproducible stratified train, validation, and test splits."""

    X_train, X_test, y_train, y_test = create_train_test_split()

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )
