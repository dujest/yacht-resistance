import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

Centre_of_Buoyancy_ix, Prismatic_Coefficient_ix, Froude_Number_ix = 0, 1, 5


class AttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_Cp_Fn_ratio=True):
        self.add_Cp_Fn_ratio = add_Cp_Fn_ratio

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(X)
        Lbc_Fn_ratio = X[:, Centre_of_Buoyancy_ix] / X[:, Froude_Number_ix]
        if self.add_Cp_Fn_ratio:
            Cp_Fn_ratio = X[:, Prismatic_Coefficient_ix] / \
                X[:, Froude_Number_ix]
            return np.c_[X, Lbc_Fn_ratio, Cp_Fn_ratio]
        else:
            return np.c_[X, Lbc_Fn_ratio]


# create a transformation pipeline for numerical attributes
yacht_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # impute the missing values
    ("attributes_adder", AttributeAdder()),         # add new attributes
    ("features_scaler", StandardScaler()),          # scale the data
])

# column names from request
columns = ["length_wl", "beam_wl", "draft", "displacement",
           "centre_of_buoyancy", "prismatic_coefficient", "velocity"]

# transformed column names for the ml model
columns_tr = ["Centre_of_Buoyancy", "Prismatic_Coefficient", "Length/Displacement_Ratio", "Beam/Draft_Ratio",
              "Length/Beam_Ratio", "Froude_Number"]


def predict_resistance(parameters, model):

    df = pd.DataFrame(parameters, columns=columns, index=[0])

    df_tr = pd.DataFrame(columns=columns_tr)

    df_tr["Length/Displacement_Ratio"] = df["length_wl"] / \
        np.power(df["displacement"], (1/3))

    df_tr["Beam/Draft_Ratio"] = df["beam_wl"] / df["draft"]

    df_tr["Length/Beam_Ratio"] = df["length_wl"] / df["beam_wl"]

    df_tr["Centre_of_Buoyancy"] = df["centre_of_buoyancy"]

    df_tr["Prismatic_Coefficient"] = df["prismatic_coefficient"]

    """
        Velocity [m/s] = Velocity [knots] * 0.51444       

                         Velocity [m/s]
        Froude Number = ----------------
                          sqrt( g * Lwl )
        
        g [m/s^2] - gravitational acceleration

        Lwl [m] - waterline length

    """

    kt_ms = 0.51444

    g = 9.81

    df_tr["Froude_Number"] = (df["velocity"] * kt_ms) / \
        np.sqrt(g * df["length_wl"])

    """
                         Resistance * 1000   
        y_predicted = ------------------------
                       displacement * rho * g

        Resistance [N] = (y_predicted * displacement * rho * g) / 1000

        rho = 1025 [kg/m^3] - seawater density

    """

    prepared_df = yacht_pipeline.fit_transform(df_tr)
    y_predicted = model.predict(prepared_df)

    rho = 1025

    resistance = y_predicted * df["displacement"] * rho * g / 1000

    return resistance.round(2).item()
