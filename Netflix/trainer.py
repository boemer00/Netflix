import requests
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from data import DataSourcing
import joblib
from termcolor import colored
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient 



MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "Netflix_experiment"

class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.3)
        self.X = X
        self.y = y
        
    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # Impute then Scale for numerical variables: 
        num_transformer = Pipeline([
        ('imputer', SimpleImputer(fill_value ='nan')),
        ('scaler', StandardScaler())])

        # Encode categorical variables
        cat_transformer = Pipeline([
        ('imputer', SimpleImputer(fill_value ='nan', strategy='constant')),
        ('OHO', OneHotEncoder(handle_unknown='ignore', sparse = False))])

        # Paralellize "num_transformer" and "One hot encoder"
        preprocessor = ColumnTransformer([
            ('num_transformer', num_transformer, ['Year','Runtime']),
            ('cat_transformer', cat_transformer, ['Rated', 'Language_binary'])],
        remainder='passthrough')

        self.pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('linear_model', LinearRegression())
            ])
    def fit_pipeline(self):
        self.pipeline = self.pipeline.fit(self.X, self.y)
        print('pipeline fitted')
        
    def evaluate(self, X_test, y_test):
        y_pred_train = self.pipeline.predict(self.X)
        mse_train = mean_squared_error(self.y, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        
        y_pred_test = self.pipeline.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        return (round(rmse_train, 3) ,round(rmse_test, 3))
        
    def predict(self, X):
        y_pred = self.pipeline.predict(X)
        return y_pred
   
    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

 # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)




if __name__ == "__main__":
    # Get and clean data
    N = 1000
    df = DataSourcing()
    df = df.load_data(N)
    df = df.data_wrangling(df)
    y = y
    X = X
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Train and save model, locally and
    trainer = Trainer(X_train, y_train)
    trainer.set_experiment_name('NetflixTeam')
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    print(f"rmse: {rmse}")
    trainer.save_model()
