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


class EntrenamintoModelo(object):
    def __init__(
        self,
        categorical_features,
        features_to_drop,
        eval_metrics,
        method="",
        path_save="outputs",
    ):

        self.categorical_features = categorical_features
        self.features_to_drop = features_to_drop
        self.eval_metrics = eval_metrics
        self.method = method
        self.path_save = path_save

    def load_data(self, table):
        df = pd.read_csv(table)
        print(f"Dataframe shape: {df.shape}")

        # validations
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
        X_train, X_test, y_train, y_test = train_test_split(df[columnas], df['trip_duration'], test_size=0.30, random_state=0)
        print(f"Train X shape: {X_train.shape}, y shape: {y_train.shape}")
        print(f"Test X shape: {X_test.shape}, y shape: {y_test.shape}")

        if self.method == "scale":
            scaled = StandardScaler()
            X_train = pd.DataFrame(
                scaled.fit_transform(X_train), columns=X_train.columns
            )
            X_test = pd.DataFrame(scaled.transform(X_test), columns=X_test.columns)
            scaler_file = os.path.join(self.path_save, "standarizers/scaler.pickle")
            pickle.dump(scaled, open(scaler_file, "wb"))
        elif self.method == 'minmax':
            scaled = MinMaxScaler()
            X_train = pd.DataFrame(
                scaled.fit_transform(X_train), columns=X_train.columns
            )
            X_test = pd.DataFrame(scaled.transform(X_test), columns=X_test.columns)
            scaler_file = os.path.join(self.path_save, "standarizers/minmaxscaler.pickle")
            pickle.dump(scaled, open(scaler_file, "wb"))

        return X_train, y_train, X_test, y_test

    def hyperparameter_tuning(self, X_train, y_train, X_test, y_test, num_trials):
        def objective(trial, train_data, test_data):
            trial.set_user_attr("objective", "regression"),
            trial.set_user_attr("metric", "None"),
            trial.set_user_attr("first_metric_only", True),
            param_grid = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=10),
                "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=10),
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction", 0.2, 0.95, step=0.1
                ),
                "objective": "regression",
                "metric": "None",
                "first_metric_only": True,
                #"n_estimators": trial.suggest_categorical("n_estimators", [1000]),
                "n_estimators": trial.suggest_int("min_data_in_leaf", 200, 1000, step=100),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100, step=10),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
                #"bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
                "bagging_freq": trial.suggest_int("num_leaves", 1, 20, step=1),
                "feature_fraction": trial.suggest_float(
                   "feature_fraction", 0.2, 0.95, step=0.1
                ),
            }
            param_grid["num_leaves"] = trial.suggest_int(
                "num_leaves",
                pow(param_grid["max_depth"], 2) - 1,
                pow(param_grid["max_depth"], 2) - 1,
            )
            eval_results = {}
            lgb.train(
                param_grid,
                train_data,
                num_boost_round=1000,
                valid_sets=[test_data],  # [train_data, test_data]
                feval=self.eval_metrics,
                early_stopping_rounds=10,
                verbose_eval=1,
                evals_result=eval_results,
                callbacks=[
                    LightGBMPruningCallback(
                        trial, self.eval_metrics[0].get_metric_name()
                    )
                ],
            )
            best_run = (
                max(eval_results["valid_0"][self.eval_metrics[0].get_metric_name()])
                if self.eval_metrics[0].is_higher_better
                else min(
                    eval_results["valid_0"][self.eval_metrics[0].get_metric_name()]
                )
            )

            return best_run

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )
        test_data = lgb.Dataset(
            X_test,
            label=y_test,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )

        direction = "maximize" if self.eval_metrics[0].is_higher_better else "minimize"
        study_filename = os.path.join(self.path_save, "hyperparameter_tuning/study.db")
        study = optuna.create_study(
            direction=direction,
            study_name="LGBM Classifier",
            storage=f"sqlite:///{study_filename}",
            load_if_exists=True,
        )
        func = lambda trial: objective(trial, train_data, test_data)
        study.optimize(func, n_trials=num_trials)

        print(
            f"Best {self.eval_metrics[0].get_metric_name()} value: {study.best_value}"
        )
        print("Best params")
        best_params = {**study.best_trial.user_attrs, **study.best_params}
        for key, value in best_params.items():
            print(f"\t{key}: {value}")

        print("Training and predicting best model:")
        self.train(X_train, y_train, X_test, y_test, best_params)

    def train(self, X_train, y_train, X_test, y_test, params=None):

        if params is None:
            params = {
                "objective": "regression",
                "metric": "None",
                "first_metric_only": True,
            }

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=self.categorical_features,
            free_raw_data=False,
        )
        test_data = lgb.Dataset(
            X_test,
            label=y_test,
            categorical_feature=self.categorical_features,
            free_raw_data=False,
        )

        eval_results = {}
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[test_data],  # [train_data, test_data]
            feval=self.eval_metrics,
            early_stopping_rounds=10,
            verbose_eval=1,
            evals_result=eval_results,
        )

        # saving artifacts
        print("Predicting model")
        now = datetime.now()
        preds = model.predict(X_test)
        print("Saving model")
        model_filename = f"models/model_{now.strftime('%d-%m-%Y_%H_%M_%S')}.txt"
        model.save_model(os.path.join(self.path_save, model_filename))
        
        feature_importances_lgb = pd.DataFrame (model.feature_importance(), 
                                                index = X_test.columns, 
                                                columns = ['importance'])
        feature_importances_lgb.sort_values('importance', ascending = False, inplace = True)
        feature_importances_lgb.to_csv(self.path_save + '/imp_variables/importancia_variables_v5.csv')
        
        
        # logging metric
        best_run = (
            max(eval_results["valid_0"][self.eval_metrics[0].get_metric_name()])
            if self.eval_metrics[0].is_higher_better
            else min(eval_results["valid_0"][self.eval_metrics[0].get_metric_name()])
        )
        print(f"Best score: {best_run}")

    def predict(self, model_file, X, y):
        model = lgb.Booster(model_file=model_file)
        print("Predicting model")
        now = datetime.now()
        preds = model.predict(X)
        res = pd.DataFrame({"pred": preds, "label": y})
        res_filename = f"preds/pred_full_v5.csv"
        res = pd.concat([res.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
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

    model = EntrenamintoModelo(
        categorical_features=categorical_features,
        features_to_drop=features_to_drop,
        eval_metrics=eval_metrics,
        method="minmax",
    )
    print("Model instantiated!")
    df = model.load_data(table="1. New york taxi.csv")
    
    X_train, y_train, X_test, y_test = model.data_prepro(df)
    print(X_train.head())
    print('y:', y_train.head())
    # params = {
    #     "objective": "binary",
    #     "metric": "None",
    #     "first_metric_only": True,
    #     "bagging_fraction": 0.5,
    #     "lambda_l1": 40,
    #     "lambda_l2": 100,
    #     "learning_rate": 0.10221,
    #     "max_depth": 8,
    #     "num_leaves": 63,
    # }
    # command to train
    #model.train(X_train, y_train, X_test, y_test, params)

    # command for hyperparameter tuning
    #model.hyperparameter_tuning(X_train, y_train, X_test, y_test, num_trials=10)

    # command to predict from saved model
    model_file = "outputs\models\model_20-05-2022_00_32_11.txt"
    model.predict(model_file, X_test, y_test)