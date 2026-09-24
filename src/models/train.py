from pathlib import Path
import joblib
import keras_tuner as kt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential

from src.pipeline.preprocessor import (
    get_features,
    get_preprocessor,
    save_preprocessor,
)


def load_and_clean_data(default_path: str = "data/raw/Food_Delivery_Time_Prediction.csv"):
    path = Path(default_path)
    
    if not path.exists():
        fallback = Path("Food_Delivery_Time_Prediction.csv")
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{path}' or '{fallback}'. "
                "Please ensure Food_Delivery_Time_Prediction.csv is inside data/raw/ or the project root."
            )

    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    df = df.drop(columns=["Order_ID", "Order_Date"], errors="ignore")
    return df


def build_tuning_model(hp, input_dim: int):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    num_layers = hp.Int("hidden", min_value=1, max_value=10, step=1)

    for i in range(num_layers):
        nodes = hp.Int(f"nodes_{i}", min_value=32, max_value=512, step=32)
        model.add(Dense(nodes, activation="relu"))

        dropout_val = hp.Float(
            f"dropout_{i}", min_value=0.0, max_value=0.8, step=0.1
        )
        if dropout_val > 0.0:
            model.add(Dropout(dropout_val))

    model.add(Dense(1))

    opt = hp.Choice("optimizer", values=["RMSprop", "Adam", "AdamW", "Adagrad"])
    model.compile(
        optimizer=opt,
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )

    return model


def main():
    df = load_and_clean_data("data/raw/Food_Delivery_Time_Prediction.csv")

    X = df.iloc[:, :-1]
    y = df["Time_taken_min"]

    # 80/20 train/test split, followed by 80/20 train/val split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Feature transformation pipeline
    num_data, ord_data, nom_data = get_features()
    features = get_preprocessor(num_data, ord_data, nom_data)

    X_train = features.fit_transform(X_train)
    X_test = features.transform(X_test)
    X_val = features.transform(X_val)

    # Save fitted preprocessor to root (ready for app.py)
    save_preprocessor(features, "preprocessor.pkl")
    print("Preprocessor fitted and saved to preprocessor.pkl")

    input_dim = X_train.shape[1]

    # Hyperparameter search via Keras Tuner
    print("Starting Keras Tuner search (20 trials)...")
    tuner = kt.RandomSearch(
        lambda hp: build_tuning_model(hp, input_dim=input_dim),
        objective="val_loss",
        max_trials=20,
        directory="Food_Delivery_Time_Prediction",
        project_name="Time_Taken_Prediction",
        overwrite=True,
    )

    tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hp)

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    print("Training final Neural Network with best hyperparameters...")
    best_model.fit(
        X_train,
        y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1,
    )

    print("Evaluating model performance on test partition...")
    y_pred = best_model.predict(X_test).flatten()
    print(f"Neural Network Test MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"Neural Network Test R2 Score: {r2_score(y_test, y_pred):.4f}")

    best_model.save("delivery_nn_model.keras")
    print("Optimal model saved successfully to delivery_nn_model.keras")


if __name__ == "__main__":
    main()