import pandas as pd

from src.config import DATA_DIR


DATASET_FILENAME = "default_of_credit_card_clients.xls"
DATASET_PATH = DATA_DIR / DATASET_FILENAME

FEATURE_COLUMNS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

TARGET_COLUMN = "default_payment_next_month"


def load_raw_data() -> pd.DataFrame:
    """Load and perform basic cleaning on the UCI credit-card dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Download the UCI dataset before loading it."
        )

    df = pd.read_excel(DATASET_PATH, engine="xlrd")
    df = df.iloc[1:].reset_index(drop=True)
    df = df.drop(columns="Unnamed: 0")
    df = df.apply(pd.to_numeric)

    df.columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    return df


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the cleaned dataset and separate features from the target."""

    df = load_raw_data()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y
