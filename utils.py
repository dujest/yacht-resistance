import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pickle

Centre_of_Buoyancy_ix, Prismatic_Coefficient_ix, Length_Displacement_Ratio_ix, Beam_Draft_Ratio_ix, Length_Beam_Ratio_ix, Froude_Number_ix = 0, 1, 2, 3, 4, 5


class AttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_Dim_Fn_ratio=False):
        self.add_Dim_Fn_ratio = add_Dim_Fn_ratio

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Lbc_Fn_ratio = X[:, Centre_of_Buoyancy_ix] / X[:, Froude_Number_ix]
        Cp_Fn_ratio = X[:, Prismatic_Coefficient_ix] / X[:, Froude_Number_ix]
        if self.add_Dim_Fn_ratio:
            LD_Fn_ratio = X[:, Length_Displacement_Ratio_ix] / \
                X[:, Froude_Number_ix]
            LB_Fn_ratio = X[:, Length_Beam_Ratio_ix] / X[:, Froude_Number_ix]
            BT_Fn_ratio = X[:, Beam_Draft_Ratio_ix] / X[:, Froude_Number_ix]
            return np.c_[X, Lbc_Fn_ratio, Cp_Fn_ratio, LD_Fn_ratio, LB_Fn_ratio, BT_Fn_ratio]
        else:
            return np.c_[X, Lbc_Fn_ratio, Cp_Fn_ratio]


with open('./scale.bin', 'rb') as f_in:
        mean = pickle.load(f_in)
        standard_deviation = pickle.load(f_in)
        f_in.close()


class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        prepared_df = (X - mean) / standard_deviation

        return prepared_df
    

yacht_pipeline = Pipeline([
    ("attributes_adder", AttributeAdder()),  # add new attributes
    ("features_scaler", Scaler()),           # scale the data
])


class ResistancePredictor:
    def __init__(self, model):
        self.model = model
        # column names from request
        self.columns = ["length_wl", "displacement", "beam_wl", "draft", "centre_of_buoyancy", "prismatic_coefficient", "velocity"]
        # transformed column names for the ml model
        self.columns_tr = ["Centre_of_Buoyancy", "Prismatic_Coefficient", "Length/Displacement_Ratio", "Beam/Draft_Ratio", "Length/Beam_Ratio", "Froude_Number"]

    def prepare_dataframe(self, parameters):
        df = pd.DataFrame(parameters, columns=self.columns, index=[0])
        df_tr = pd.DataFrame(columns=self.columns_tr)
        df_tr["Length/Displacement_Ratio"] = df["length_wl"] / np.power(df["displacement"], (1/3))
        df_tr["Beam/Draft_Ratio"] = df["beam_wl"] / df["draft"]
        df_tr["Length/Beam_Ratio"] = df["length_wl"] / df["beam_wl"]
        df_tr["Centre_of_Buoyancy"] = df["centre_of_buoyancy"]
        df_tr["Prismatic_Coefficient"] = df["prismatic_coefficient"]
        return df, df_tr
    
    """
        Velocity [m/s] = Velocity [knots] * 0.51444       

                         Velocity [m/s]
        Froude Number = ----------------
                          sqrt( g * Lwl )
        
        g [m/s^2] - gravitational acceleration

        Lwl [m] - waterline length

    """
    def calculate_froude_number(self, df, df_tr):
        kt_ms = 0.51444
        g = 9.81
        df_tr["Froude_Number"] = (df["velocity"] * kt_ms) / np.sqrt(g * df["length_wl"])
        return df_tr
    
    """
                         Resistance * 1000   
        y_predicted = ------------------------
                       displacement * rho * g

        Resistance [N] = (y_predicted * displacement * rho * g) / 1000

        rho = 1025 [kg/m^3] - seawater density

    """
    def predict_resistance(self, parameters):
        df, df_tr = self.prepare_dataframe(parameters)
        df_tr = self.calculate_froude_number(df, df_tr)
        prepared_df = yacht_pipeline.transform(df_tr.to_numpy())
        y_predicted = self.model.predict(prepared_df)
        rho = 1025
        g = 9.81
        resistance = y_predicted * df["displacement"] * rho * g / 1000
        return resistance.round(2).item()
