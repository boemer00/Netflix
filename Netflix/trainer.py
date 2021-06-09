import pandas as pd
import joblib
from termcolor import colored
import mlflow
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



        
#     def set_pipeline(self):
#         """defines the pipeline as a class attribute"""
#         pass
    
#     def run(self):
#         """set and train the pipeline"""
#         pass
    
#     def evaluate(self, X_test, y_test):
#         """evaluates the pipeline on df_test and return the RMSE"""
#         pass

# if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
  #  print('TODO')
    
    
    


MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "first_experiment"


class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
    
    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pass
        


    def set_pipeline(self):
         """defines the pipeline as a class attribute"""
        # dist_pipe = Pipeline([
        #     ('dist_trans', DistanceTransformer()),
        #     ('stdscaler', StandardScaler())
        # ])
        # time_pipe = Pipeline([
        #     ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        #     ('ohe', OneHotEncoder(handle_unknown='ignore'))
        # ])
        # preproc_pipe = ColumnTransformer([
        #     ('distance', dist_pipe, [
        #         "pickup_latitude",
        #         "pickup_longitude",
        #         'dropoff_latitude',
        #         'dropoff_longitude'
        #     ]),
        #     ('time', time_pipe, ['pickup_datetime'])
        # ], remainder="drop")
        # 
         self.pipeline = Pipeline([
        #    ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

    def run(self):
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return round(rmse, 2)

    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

   


if __name__ == "__main__":
    # Get and clean data
    N = 10000
    df = get_data(nrows=N)
    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Train and save model, locally and
    trainer = Trainer(X=X_train, y=y_train)
    trainer.set_experiment_name('xp2')
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    print(f"rmse: {rmse}")
    trainer.save_model()
