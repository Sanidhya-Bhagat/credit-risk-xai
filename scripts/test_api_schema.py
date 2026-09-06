from api.schemas import CreditRiskInput


customer = CreditRiskInput(
    LIMIT_BAL=50000,
    SEX=2,
    EDUCATION=2,
    MARRIAGE=1,
    AGE=35,
    PAY_0=0,
    PAY_2=0,
    PAY_3=0,
    PAY_4=0,
    PAY_5=0,
    PAY_6=0,
    BILL_AMT1=10000,
    BILL_AMT2=9000,
    BILL_AMT3=8000,
    BILL_AMT4=7000,
    BILL_AMT5=6000,
    BILL_AMT6=5000,
    PAY_AMT1=1000,
    PAY_AMT2=1000,
    PAY_AMT3=1000,
    PAY_AMT4=1000,
    PAY_AMT5=1000,
    PAY_AMT6=1000,
)

print("=== API SCHEMA TEST ===")
print()
print("Valid customer accepted successfully.")
print(f"Number of input features: {len(customer.model_dump())}")