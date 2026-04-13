import pandas as pd  
import numpy as np
from xgboost import XGBRegressor  # model za regresiju
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. load data

train = pd.read_parquet("block1_train.parquet")
test = pd.read_parquet("block3_test.parquet")

#print(train.columns.tolist())

print("Train shape:", train.shape)
print("Test shape:", test.shape)


# 2. TARGET COLUMNS

price_cols = [col for col in train.columns if "price" in col]
print("Price columns:", price_cols)

# 3. PREPROCESSING FUNCTION

def preprocess(df):
    df = df.copy()

    if "contractor_birthdate" in df.columns:
        dob = pd.to_datetime(df["contractor_birthdate"], errors="coerce", dayfirst=True)
        df["driver_age"] = 2026 - dob.dt.year
        df["driver_age"] = df["driver_age"].clip(lower=16, upper=100)

    if "vehicle_age" in df.columns:
        df["car_age"] = pd.to_numeric(df["vehicle_age"], errors="coerce").clip(lower=0, upper=40)
    elif "vehicle_first_registration_date" in df.columns:
        reg = pd.to_datetime(df["vehicle_first_registration_date"], errors="coerce", dayfirst=True)
        df["car_age"] = 2026 - reg.dt.year
        df["car_age"] = df["car_age"].clip(lower=0, upper=40)

    if "claim_free_years" in df.columns:
        df["claim_free_years"] = pd.to_numeric(df["claim_free_years"], errors="coerce")
        df["is_risky_driver"] = (df["claim_free_years"] < 0).astype(int)

    if "vehicle_power" in df.columns and "vehicle_net_weight" in df.columns:
        safe_weight = pd.to_numeric(df["vehicle_net_weight"], errors="coerce").replace(0, np.nan) + 1
        df["power_to_weight"] = pd.to_numeric(df["vehicle_power"], errors="coerce") / safe_weight

    if "vehicle_value_new" in df.columns and "car_age" in df.columns:
        df["value_per_year"] = pd.to_numeric(df["vehicle_value_new"], errors="coerce") / (df["car_age"] + 1)

    if "vehicle_planned_annual_mileage" in df.columns and "car_age" in df.columns:
        df["mileage_per_year"] = pd.to_numeric(df["vehicle_planned_annual_mileage"], errors="coerce") / (df["car_age"] + 1)

    if "coverage" in df.columns:
        df["coverage"] = df["coverage"].astype("category")

    deductible_cols = [c for c in df.columns if "deductible" in c.lower()]
    if deductible_cols:
        df["deductible_min"] = df[deductible_cols].min(axis=1)
        df["deductible_max"] = df[deductible_cols].max(axis=1)
        df["deductible_mean"] = df[deductible_cols].mean(axis=1)
        df["deductible_std"] = df[deductible_cols].std(axis=1)

    # ── NOVI FEATURES based on top importance ─────────

    # 1. young driver + expensive car
    if "driver_age" in df.columns and "vehicle_power" in df.columns:
        power = pd.to_numeric(df["vehicle_power"], errors="coerce")
        df["young_driver_high_power"] = (
            (df["driver_age"] < 25) & (power > 100)
        ).astype(int)

    # 2. risky driver+pricey car
    if "is_risky_driver" in df.columns and "vehicle_value_new" in df.columns:
        value = pd.to_numeric(df["vehicle_value_new"], errors="coerce")
        df["risky_driver_expensive_car"] = (
            (df["is_risky_driver"] == 1) & (value > value.quantile(0.75))
        ).astype(int)

    # 3. claim_free_years * driver_age
    if "claim_free_years" in df.columns and "driver_age" in df.columns:
        df["age_x_risk"] = df["driver_age"] * df["claim_free_years"].fillna(0)

    # 4. Car age * value
    if "car_age" in df.columns and "vehicle_value_new" in df.columns:
        value = pd.to_numeric(df["vehicle_value_new"], errors="coerce")
        df["old_expensive_car"] = df["car_age"] * value

    # 5. Second driver features
    if "second_driver_birthdate" in df.columns:
        dob2 = pd.to_datetime(df["second_driver_birthdate"], errors="coerce", dayfirst=True)
        df["second_driver_age"] = (2026 - dob2.dt.year).clip(16, 100)
        df["has_second_driver"] = df["second_driver_age"].notnull().astype(int)

    if "second_driver_claim_free_years" in df.columns:
        df["second_driver_claim_free_years"] = pd.to_numeric(
            df["second_driver_claim_free_years"], errors="coerce"
        )
        df["second_is_risky"] = (df["second_driver_claim_free_years"] < 0).astype(int)

    # combined_risk for both drivers
    if "claim_free_years" in df.columns and "second_driver_claim_free_years" in df.columns:
        df["combined_risk"] = (
            df["claim_free_years"].fillna(0) + df["second_driver_claim_free_years"].fillna(0)
        )

    # 6. Value per power
    if "vehicle_value_new" in df.columns and "vehicle_power" in df.columns:
        safe_power = pd.to_numeric(df["vehicle_power"], errors="coerce").replace(0, np.nan) + 1
        df["value_per_power"] = pd.to_numeric(df["vehicle_value_new"], errors="coerce") / safe_power

    # 7. Urban
    if "postal_code_urban_category" in df.columns and "postal_code_average_property_value" in df.columns:
        df["urban_x_property"] = (
            pd.to_numeric(df["postal_code_urban_category"], errors="coerce") *
            pd.to_numeric(df["postal_code_average_property_value"], errors="coerce")
        )

    # 8. Criminal * driver age
    if "municipality_crimes_per_1000" in df.columns and "driver_age" in df.columns:
        crimes = pd.to_numeric(df["municipality_crimes_per_1000"], errors="coerce")
        df["young_in_crime_area"] = (df["driver_age"] < 30) * crimes

    return df

# 4. APPLY PREPROCESSING

train = preprocess(train)
test = preprocess(test)

# 5. SPLIT FEATURES / TARGET

X = train.drop(columns=price_cols)
X_test = test.copy()


# 6. HANDLE CATEGORICALS - Label Encoding


from sklearn.preprocessing import LabelEncoder

cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col].astype(str), X_test[col].astype(str)])
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# 7. HANDLE MISSING - only num

num_cols = X.select_dtypes(include=[np.number]).columns

for col in num_cols:
    median = X[col].median()
    X[col] = X[col].fillna(median)
    X_test[col] = X_test[col].fillna(median)

# 8. TRAIN MODELS

models = {}
scores = {}

# define weights
feature_weights_map = {
    "driver_age": 5,
    "claim_free_years": 5,
    "municipality": 4,
    "payment_frequency": 4,
    "vehicle_net_weight": 3,
    "coverage": 3,
    "vehicle_value_new": 3,
    "vehicle_fuel_type": 2,
    "vehicle_model": 2,
    "young_driver_high_power": 2,
    "value_per_year": 2,
    "vehicle_power": 2,
}

for col in price_cols:
    print(f"\n🔹 Training {col}...")

    mask = train[col].notnull()
    y = train.loc[mask, col]
    X_temp = X.loc[mask]

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y, test_size=0.2, random_state=42
    )


    feature_weights = np.array([
        feature_weights_map.get(c, 1) for c in X_train.columns
    ], dtype=float)

    model = LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=127,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        feature_fraction_bynode=0.8,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        feature_name=list(X_train.columns),
        callbacks=[early_stopping(100), log_evaluation(500)]
    )

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"{col} MAE: {mae:.2f}")

    models[col] = model
    scores[col] = mae


# 9. PREDICTIONS

preds_df = pd.DataFrame()
preds_df["quote_id"] = test["quote_id"]

for col in price_cols:
    preds_df[col] = models[col].predict(X_test)


# 10. SAVE SUBMISSION


preds_df[price_cols] = preds_df[price_cols].round(2)


preds_df.to_csv("result2.csv", sep=";", index=False)

print("\nDONE! File saved as submission_block2.csv")