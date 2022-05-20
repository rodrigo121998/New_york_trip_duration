import os
import pandas as pd
import numpy as np
import sys
pd.set_option("display.float_format", lambda x: "%.3f" % x)
from sqlalchemy import create_engine
from datetime import datetime
import optuna
from optuna.integration import LightGBMPruningCallback
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
#sys.path.insert(0, "D:\Datos de Usuario\rosorio\Documents\Python Scripts\Rappi\Prueba Técnica Python_202205")
from metrics import R2_score,RMSLE,RMSE,MAE

import warnings

warnings.filterwarnings("ignore")


class PrediccionModelo(object):
    def __init__(
        self,
        categorical_features,
        features_to_drop,
        method="",
        path_save="outputs",
    ):
        self.categorical_features = categorical_features
        self.features_to_drop = features_to_drop
        self.method = method
        self.path_save = path_save

    def load_data(self, table):
        df = pd.read_sql_query(table)
        print(f"Dataframe shape: {df.shape}")

        # preprocessing
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

        df['pickup_day'] = df['pickup_datetime'].dt.day
        df['pickup_month'] = df['pickup_datetime'].dt.month
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_min'] = df['pickup_datetime'].dt.minute
        df['pickup_weekday'] = df['pickup_datetime'].dt.weekday

        df['dropoff_min'] = df['dropoff_datetime'].dt.minute
        df['dropoff_hour'] = df['dropoff_datetime'].dt.hour
        df['log_trip_duration']=np.log(df['trip_duration'])
        df=pd.get_dummies(df,prefix=['vendor_id','store_and_fwd_flag'], columns = ['vendor_id','store_and_fwd_flag'], drop_first=True)

        return df

    def data_prepro(self, data):
        columnas = [x for x in data.columns if x not in self.features_to_drop]
        id=data['id'] 
        X_test = data[columnas]        
        print(f"Data shape: {X_test.shape}")

        if self.method == "scale":
            scaled = StandardScaler()
            X_test = pd.DataFrame(scaled.transform(X_test), columns=X_test.columns)
        elif self.method == 'minmax':
            scaled = MinMaxScaler()
            X_test = pd.DataFrame(scaled.transform(X_test), columns=X_test.columns)

        return X_test,id

    def predict(self, model_file, X,nro_documento_ls):
        model = lgb.Booster(model_file=model_file)
        print("Predicting model")
        now = datetime.now()
        X['score'] = model.predict(X)
        doc = pd.DataFrame({"id": id})
        res_filename = f"preds/pred_full_v5.csv"
        res = pd.concat([doc.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        res.to_csv(os.path.join(self.path_save, res_filename),sep=';')


if __name__ == "__main__":

    categorical_features = [
        "pickup_day", 
        "pickup_month",
        "pickup_hour",
        "pickup_min",
        "pickup_weekday",
        "vendor_id_2",
        "store_and_fwd_flag_Y",
    ]
    features_to_drop = [
        'log_trip_duration',
        'trip_duration',
        'id',
        'pickup_datetime',
        'dropoff_datetime',
        'dropoff_min',
        'dropoff_hour'
    ]

    eval_metrics = [
        RMSE(metric_name="RMSE",is_higher_better=False),
        MAE(metric_name="MAE",is_higher_better=False),
        R2_score(metric_name="r2_score",is_higher_better=True),
    ]

    model = PrediccionModelo(
        categorical_features=categorical_features,
        features_to_drop=features_to_drop,
        eval_metrics=eval_metrics,
        method="minmax",
    )
    print("Model instantiated!")
    df = model.load_data(table="1. New york taxi.csv")
    
    data_vars,id = model.data_prepro(df)
    
    # command to predict from saved model
    model_file = "model_20-05-2022_00_32_11.txt"
    model.predict(model_file, data_vars,id)