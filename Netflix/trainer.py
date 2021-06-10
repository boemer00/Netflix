class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.3)
        
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
        self.pipeline = self.pipeline.fit(self.X_train, self.y_train)
        print('pipeline fitted')
        
    def evaluate(self):
        y_pred_train = self.pipeline.predict(self.X_train)
        mse_train = mean_squared_error(self.y_train, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        
        y_pred_test = self.pipeline.predict(self.X_test)
        mse_test = mean_squared_error(self.y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        return (round(rmse_train, 3) ,round(rmse_test, 3))
        
    def predict(self, X):
        y_pred = self.pipeline.predict(X)
        return y_pred
   


# if __name__ == "__main__":
#     # Get and clean data
#     N = 10000
#     df = get_data(nrows=N)
#     df = clean_data(df)
#     y = df["fare_amount"]
#     X = df.drop("fare_amount", axis=1)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#     # Train and save model, locally and
#     trainer = Trainer(X=X_train, y=y_train)
#     trainer.set_experiment_name('xp2')
#     trainer.run()
#     rmse = trainer.evaluate(X_test, y_test)
#     print(f"rmse: {rmse}")
#     trainer.save_model()
