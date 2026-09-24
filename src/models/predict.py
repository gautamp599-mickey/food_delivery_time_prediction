import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.pipeline.preprocessor import load_preprocessor


class DeliveryPredictor:

    def __init__(
        self,
        model_path="delivery_nn_model.keras",
        preprocessor_path="preprocessor.pkl",
    ):
        self.preprocessor = load_preprocessor(preprocessor_path)
        self.model = load_model(model_path)

    def predict(self, input_df: pd.DataFrame) -> float:
        processed = self.preprocessor.transform(input_df)
        prediction = self.model.predict(processed, verbose=0)[0][0]
        return float(np.round(prediction))