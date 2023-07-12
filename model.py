import numpy as np
from sklearn.pipeline import Pipeline
from utils import AttributeAdder, Scaler, ResistancePredictor
import pandas as pd

def predict_resistance(parameters, model):

    resistance_predictor = ResistancePredictor(model)

    resistance = resistance_predictor.predict_resistance(parameters)

    return resistance
