RISK_BANDS = [
    (0.00, 0.10, "0–10%"),
    (0.10, 0.20, "10–20%"),
    (0.20, 0.30, "20–30%"),
    (0.30, 0.40, "30–40%"),
    (0.40, 0.50, "40–50%"),
    (0.50, 0.60, "50–60%"),
    (0.60, 0.70, "60–70%"),
    (0.70, 0.80, "70–80%"),
    (0.80, 0.90, "80–90%"),
    (0.90, 1.01, "90–100%"),
]


def get_risk_band(probability: float) -> str:
    """Assign a probability to its corresponding risk band."""

    for lower, upper, label in RISK_BANDS:
        if lower <= probability < upper:
            return label

    raise ValueError(f"Probability outside valid range: {probability}")