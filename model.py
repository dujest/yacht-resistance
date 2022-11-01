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
        Lbc_Fn_ratio = X[:, Centre_of_Buoyancy_ix] / X[:, Froude_Number_ix] 
        if self.add_Cp_Fn_ratio:
            Cp_Fn_ratio = X[:, Prismatic_Coefficient_ix] / X[:, Froude_Number_ix] 
            return np.c_[X, Lbc_Fn_ratio, Cp_Fn_ratio]
        else:
            return np.c_[X, Lbc_Fn_ratio]


# create a transformation pipeline for numerical attributes
yacht_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")), # impute the missing values
            ("attributes_adder", AttributeAdder()),        # add new attributes
            ("features_scaler", StandardScaler()),         # scale the data
        ])

# column names from request
columns = ["Length", "Beam", "Draft", "Displacement", 
           "Centre_of_Buoyance", "Prismatic_Coefficient", "Froude_Number"]

# transformed column names for the ml model
columns_tr = ["Centre_of_Buoyancy", "Prismatic_Coefficient", "Length/Displacement_Ratio", "Beam/Draft_Ratio", 
              "Length/Beam_Ratio", "Froude_Number"]

def predict_resistance(parameters, model):

    df = pd.DataFrame(parameters, columns=columns)

    df_tr = pd.DataFrame(columns=columns_tr)

    df_tr["Length/Displacement_Ratio"] = df["Length"] / df["Displacement"]

    df_tr["Beam/Draft_Ratio"] = df["Beam"] / df["Draft"]
    
    df_tr["Length/Beam_Ratio"] = df["Length"] / df["Beam"]

    df_tr["Centre_of_Buoyancy"] = df["Centre_of_Buoyancy"]

    df_tr["Prismatic_Coefficient"] = df["Prismatic_Coefficient"]

    df_tr["Prismatic_Coefficient"] = df["Prismatic_Coefficient"]

    prepared_df = yacht_pipeline.transform(df_tr)
    y_predicted = model.predict(prepared_df)
    return y_predicted
