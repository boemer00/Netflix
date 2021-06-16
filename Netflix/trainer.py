from types import FunctionType
import joblib
import requests
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split

from Netflix.params import MLFLOW_URI, EXPERIMENT_NAME
from Netflix.data import load_data
from Netflix.encoders import CleanRuntimeEncoder, CleanLanguageEncoder, CleanCountryEncoder,\
                             CleanReleasedEncoder, CleanRatedEncoder, CleanAgeEncoder
import mlflow
from mlflow.tracking import MlflowClient 

from xgboost import XGBRegressor
from termcolor import colored
from memoized_property import memoized_property


MLFLOW_URI='https://mlflow.lewagon.co/'

class Trainer(object):
    ESTIMATOR = "Linear"
    EXPERIMENT_NAME = "[UK] [London] [PDR] netflix"
    
    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.3)
        self.kwargs = kwargs
        self.X = X
        self.y = y
        
        # for MLFlow
        self.experiment_name = EXPERIMENT_NAME
        
        
    def get_estimator(self):
        estimator = self.kwargs.get('estimator', self.ESTIMATOR)
        if estimator == 'Lasso':
            model = Lasso() # (alpha = 0) == Linear Regression
            self.model_params = {'alpha':[1, 2, 3, 4, 5, 10],
                                 'fit_intercept': [True, False],
                                 'max_iter':[1000, 2000, 5000, 10000],
                                 'random_state': 0}
        
        elif estimator == 'Ridge':
            model = Ridge()
            self.model_params = {'alpha': [0, 0.2, 0.4, 0.6, 0.8],  # (alpha = 1) == Linear Regression
                                 'normalize':[True, False],
                                 'solver':['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                                 'random_state': 0} 
        
        elif estimator == 'Linear':
            model = LinearRegression()
            self.model_params = {'fit_intercept': [True, False],
                                 'normalize': [True, False]}
        
        elif estimator == 'KNN':
            model = KNeighborsRegressor()
            self.model_params = {'n_neighbors':[5, 10, 15, 20, 30, 50],
                                 'weights': ['uniform', 'distance'],
                                 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
        
        elif estimator == 'Bagging':
            model = BaggingRegressor(
                base_estimator=GradientBoostingRegressor(
                    loss='huber',
                    learning_rate=0.05,
                    max_features=14,
                    n_estimators=120,
                    max_depth=7),
                n_estimators=3,
                bootstrap=True)
        
        elif estimator == 'Ada':
            model = AdaBoostRegressor(
                base_estimator=GradientBoostingRegressor(
                    loss='huber',
                    learning_rate=0.05,
                    max_features=14,
                    n_estimators=120,
                    max_depth=7),
                n_estimators=100, 
                loss='linear')                       
        
        elif estimator == 'Stacking':
            temp_=[('gbr', GradientBoostingRegressor(
                    loss='huber',
                    learning_rate=0.05,
                    max_features=14,
                    n_estimators=120,
                    max_depth=7))]  
            model = StackingRegressor(
                estimators=temp_,
                final_estimator=XGBRegressor(
                    max_depth=26,
                    n_estimators=155,
                    learning_rate=0.05,
                    gamma=1))
        
        elif estimator == 'Voting':
            model = VotingRegressor([
                ('gbr', GradientBoostingRegressor(
                    loss='huber',
                    learning_rate=0.05,
                    max_features=14,
                    n_estimators=120,
                    max_depth=7)),
                ('xgb', XGBRegressor(
                    max_depth=26,
                    n_estimators=155,
                    learning_rate=0.05,
                    gamma=1))])    
        
        elif estimator == 'GBR':
            model = GradientBoostingRegressor()
            self.model_params = {'loss': ['ls', 'huber'],
                                 'learning_rate': [1.0, 0.1, 0.05, 0.01],
                                 'max_features': [5, 7, 9, 10, 12],
                                 'n_estimators': [100, 300, 500, 1000],
                                 'random_state': 0,
                                 'max_depth' : [int(x) for x in np.linspace(2, 8, num=4)]}
        
        elif estimator == 'RandomForest':
            model = RandomForestRegressor()
            self.model_params = {'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
                                 'max_features': [5, 7, 9, 10, 12],
                                 'n_jobs': -1,
                                 'max_depth' : [int(x) for x in np.linspace(2, 8, num=4)]}
        
        elif estimator == 'xgboost':
            model = XGBRegressor(objective='reg:squarederror', n_jobs=-1, max_depth=10,
                                 learning_rate=0.05, gamma=3)
            self.model_params = {'max_depth': range(2, 40, 2),
                                 'n_estimators': range(60, 220, 40),
                                 'learning_rate': [0.5, 0.1, 0.05, 0.01, 0.001],
                                 'gamma': [1, 3, 5]}

        else:
            model = Lasso()
        estimator_params = self.kwargs.get('estimator_params', {})
        self.mlflow_log_param('estimator', estimator)
        model.set_params(**estimator_params)
        print(colored(model.__class__.__name__, 'red'))
        return model
        
        
    def set_experiment_name(self, experiment_name):
        """ defines the experiment name for MLFlow """
        self.experiment_name = experiment_name
    
    
    def set_pipeline(self):
        """ defines the pipeline as a class attribute """
        # feature engineering pipeline blocks
        feateng_steps = self.kwargs.get('feateng', ['runtime', 'country', 'language',
                                                    'genre', 'age', 'rated', 'released',
                                                    'writer', 'director', 'actors', 'production'])
        
        pipe_runtime_features = Pipeline([
            ('runtime', SimpleImputer(strategy='constant', fill_value="0")),
            ('runtime_encoder', CleanRuntimeEncoder()),
            ('runtime_scaler', StandardScaler())])
        
        pipe_country_features = Pipeline([
            ('country', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('country_encoder', CleanCountryEncoder())])
        
        pipe_language_features = Pipeline([
            ('language', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('language_encoder', CleanLanguageEncoder())])
        
        pipe_genre_features = Pipeline([
            ('genre', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('genre_transformer', FunctionTransformer(np.reshape, kw_args={'newshape':-1})), 
            ('genre_vectorizer', CountVectorizer(token_pattern='[a-zA-Z][a-z -]+', max_features=10))])
        
        pipe_age_features = Pipeline([
            ('age', SimpleImputer(strategy='median')),
            ('age_enconder', CleanAgeEncoder())])
        
        pipe_rated_features = Pipeline([
            ('rated', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('rated_encoder', CleanRatedEncoder()),
            ('rated_ohe', OneHotEncoder(handle_unknown='ignore'))])
        
        pipe_released_features = Pipeline([
            ('released', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('released_encoder', CleanReleasedEncoder()),
            ('released_ohe', OneHotEncoder(handle_unknown='ignore'))])

        pipe_writer_features = Pipeline([
            ('writer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('writer_transformer', FunctionTransformer(np.reshape, kw_args={'newshape': -1})), 
            ('writer_vectorizer', CountVectorizer(token_pattern='[a-zA-Z][a-z -]+', max_features=10))])
        
        pipe_director_features = Pipeline([
            ('director', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('director_transformer', FunctionTransformer(np.reshape, kw_args={'newshape': -1})), 
            ('director_vectorizer', CountVectorizer(token_pattern='[a-zA-Z][a-z -]+', max_features=10))])
        
        pipe_actors_features = Pipeline([
            ('actors', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('actors_transformer', FunctionTransformer(np.reshape, kw_args={'newshape': -1})), 
            ('actors_vectorizer', CountVectorizer(token_pattern='[a-zA-Z][a-z -]+', max_features=10))])
        
        pipe_production_features = Pipeline([
            ('production', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('production_transformer', FunctionTransformer(np.reshape, kw_args={'newshape': -1})), 
            ('production_vectorizer', CountVectorizer(token_pattern='[a-zA-Z][a-z -]+', max_features=10))])
        
        
        # define default feature engineering blocks
        feateng_blocks = [
            ('runtime', pipe_runtime_features, ['Runtime']),
            ('country', pipe_country_features, ['Country']),
            ('genre', pipe_genre_features, ['Genre']),
            ('age', pipe_age_features, ['Year']),
            ('rated', pipe_rated_features, ['Rated']),
            ('released', pipe_released_features, ['Released']),
            ('writer', pipe_writer_features, ['Writer']),
            ('director', pipe_director_features, ['Director']),
            ('actors', pipe_actors_features, ['Actors']),
            ('language', pipe_language_features, ['Language']),
            ('production', pipe_production_features, ['Production'])
        ]
        
        # filter out some blocks according to input parameters
        for block in feateng_blocks:
            if block[0] not in feateng_steps:
                feateng_blocks.remove(block)

        features_encoder = ColumnTransformer(feateng_blocks, n_jobs=None, remainder='drop')

        self.pipeline = Pipeline(steps=[
            ('features', features_encoder),
            ('rgs', self.get_estimator())])

    ## grid search here!

    def run(self):
        self.set_pipeline()
        self.mlflow_log_param('model', 'Linear')
        self.pipeline.fit(self.X, self.y)
        print('pipeline fitted')
        
    # def fit_pipeline(self):
    #     self.pipeline = self.pipeline.fit(self.X, self.y)
    #     print('pipeline fitted')
        
    def evaluate(self, X_test, y_test):
        """ evaluates the pipeline on X and return the RMSE """
        y_pred_train = self.pipeline.predict(self.X)
        mse_train = mean_squared_error(self.y, y_pred_train)
        rmse_train = np.sqrt(mse_train)
    
        self.mlflow_log_metric('rmse_train', rmse_train)
        
        y_pred_test = self.pipeline.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        self.mlflow_log_metric('rmse_test', rmse_test)
        
        return (round(rmse_train, 3) ,round(rmse_test, 3))
        
    def predict(self, X):
        y_pred = self.pipeline.predict(X)
        return y_pred
   
    def save_model(self):
        """ save the model into a .joblib format """
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored('model.joblib saved locally', 'green'))

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

###--------------------------------------

if __name__ == "__main__":
    # store the data in a DataFrame
    N = 8000
    df = load_data(N)
        
    # set X and y
    y = df.avg_review_score
    X = df[['Year','Runtime', 'Rated', 'Country', 'Genre', 'Language',
            'Released', 'Writer', 'Director', 'Actors', 'Production']]
    
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # train model
    estimators = ['Bagging', 'Ada', 'Stacking', 
                  'Voting', 'GBR', 'xgboost']  
    # 'Linear', 'Lasso', 'Ridge', 'KNN',
    # 'RandomForest'
    for estimator in estimators:
        params = {'estimator': estimator,
                  'feateng': ['runtime', 'country', 'genre',
                              'age', 'language', 'released',
                              'rated', 'writer', 'director',
                              'actors', 'production']}
        
        trainer = Trainer(X_train, y_train, **params)
        trainer.set_experiment_name(EXPERIMENT_NAME)
        trainer.run()
    
        # evaluate the pipeline
        rmse = trainer.evaluate(X_test, y_test)
        print(f"rmse: {rmse}")
        
        # save model locally
        trainer.save_model()
        