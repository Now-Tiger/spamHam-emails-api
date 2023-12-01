#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import joblib
from typing import Optional

from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from warnings import filterwarnings
filterwarnings('always')
filterwarnings('ignore')


class PredictionInput(BaseModel):
    text: str

    def __len__(self) -> int:
        return len(self.text)


class PredictionOutput(BaseModel):
    label: int
    

class EmailDetectionModel:
    """ 
    Loads model and Makes prediction
    """
    model: Optional[Pipeline]
    targets: Optional[list[str]]

    def load_model(self) -> None:
        """ Load the model """
        model_file = os.path.join(os.path.dirname(__file__), 'email_phishing_detection.0.1.0.joblib')
        loaded_model: tuple[Pipeline, list[str]] = joblib.load(model_file)
        model, targets = loaded_model
        self._model = model
        self._targets = targets
        self._vectorizer = joblib.load('vectorizer.joblib')

    async def predict(self, input: PredictionInput) -> PredictionOutput:
        """ Runs a prediction on given input string.
            * if input is empty raise an HTTP 404 error
            * else run prediction
        """
        if input.text == None or len(input.text) < 1:
            raise HTTPException(status_code=404, detail="Empty input string provided.")
        else:
            transformed_text = self._vectorizer.transform([input.text])
            predictions = self._model.predict(transformed_text)
            label = predictions
            return PredictionOutput(label=label)
