import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def get_features():
    """Returns the exact nominal, ordinal, and numerical column lists."""
    nom_cols = [
        "Day_of_Week",
        "Weather",
        "Pickup_Zone",
        "Dropoff_Zone",
        "Vehicle_Type",
        "Cuisine_Type",
    ]
    ord_cols = [
        "Restaurant_Load",
        "Delivery_Distance_Category",
        "Traffic_Level",
        "Delivery_Priority",
    ]
    num_cols = [
        "Order_Hour",
        "Is_Weekend",
        "Is_Festival",
        "Rider_Experience_Years",
        "Rider_Rating",
        "Restaurant_Rating",
        "Order_Items",
        "Preparation_Time_Min",
        "Road_Distance_km",
        "Number_of_Signals",
        "Average_Speed_kmph",
    ]
    return num_cols, ord_cols, nom_cols


def get_preprocessor(numerical_data, ordinal_data, nominal_data):
    """Builds the exact ColumnTransformer from your notebook."""
    num_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="mean")),
            ("scaling", StandardScaler()),
        ]
    )

    ord_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encoding",
                OrdinalEncoder(
                    categories=[
                        ["Low", "Medium", "High"],
                        ["Short", "Medium", "Long"],
                        ["Low", "Moderate", "High", "Severe"],
                        ["Normal", "Priority", "VIP"],
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
            ("scaling", StandardScaler()),
        ]
    )

    nom_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encoding",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    features = ColumnTransformer(
        transformers=[
            ("Numerical Pipeline", num_pipeline, numerical_data),
            ("Ordinal Pipeline", ord_pipeline, ordinal_data),
            ("Nominal Pipeline", nom_pipeline, nominal_data),
        ]
    )

    return features


def save_preprocessor(preprocessor, path: str = "preprocessor.pkl"):
    joblib.dump(preprocessor, path)


def load_preprocessor(path: str = "preprocessor.pkl"):
    return joblib.load(path)