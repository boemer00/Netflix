from os import X_OK
import re
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin


def load_data(n):
    """ upload the df_train.csv """
    df = pd.read_csv('raw_data/df_train.csv', nrows=n)
    return df


def drop_columns(df):
    """ drop irrelevant columns and rows """
    df = df.drop(columns=['title', 'year', 'Awards', 'Poster',
                          'Metascore', 'DVD', 'BoxOffice',
                          'Internet Movie Database', 'totalSeasons', 
                          'imdbVotes','Website', 'Response',
                          'Production', 'Metacritic', 'Ratings'])
    df = df.dropna(subset = ['Actors', 'Director', 'Writer', 'Language'])


class CleanRuntimeEncoder(BaseEstimator, TransformerMixin):
<<<<<<< HEAD
    # clean runtime encoder(CustomEncoder):
=======
    """ class clean runtime encoder """
>>>>>>> 383e56073a0b07b1fa74c5b6804d45e0b86fa24f
    def __init__(self):
        pass

    def replace_strings(self, row): 
        # regex and replace str formats
        x = row[0]
        counter = 0
        if 'h' in x:
            counter += int(x[0]) * 60
            x = re.sub('.*h', '', x).strip()
        x = x.replace('min', '').replace(',','').strip()
        counter += int(x)
        return [counter]
    
    def transform(self, x, y=None):
        final = np.array([self.replace_strings(row) for row in x])
        return final
        
    def fit(self, x, y=None):
        return self


class CleanLanguageEncoder(BaseEstimator, TransformerMixin):
<<<<<<< HEAD
    # clean language encoder(CustomEncoder):
=======
    """ class clean language encoder """
>>>>>>> 383e56073a0b07b1fa74c5b6804d45e0b86fa24f
    def __init__(self):
        pass
        
    def include_english(self, row):
        # replace with other frequent values
        x = row[0]
        if 'english' in x.lower():
            return [1]
        return [0]
            
    def transform(self, x, y=None):
        final = np.array([self.include_english(row) for row in x])
        return final
        
    def fit(self, x, y=None):
        return self


class CleanCountryEncoder(BaseEstimator, TransformerMixin):
<<<<<<< HEAD
    # clean country encoder(CustomEncoder):
=======
    """ class clean country encoder """
>>>>>>> 383e56073a0b07b1fa74c5b6804d45e0b86fa24f
    def __init__(self):
        pass
        
    def include_us(self, row):
        # replace with other frequent values
        x = row[0]
        usa = ['United States', 'USA']
        for name in usa:
            if name in x:
                return [1]
        return [0]
            
    def transform(self, x, y=None):
        final = np.array([self.include_us(row) for row in x])
        return final
        
    def fit(self, x, y=None):
        return self

<<<<<<< HEAD
=======
    
# --> NLP as a future project/next step
# class CleanPlotEncoder(BaseEstimator, TransformerMixin):
#     # class clean plot encoder(CustomEncoder):
#     def __init__(self):
#         pass
        
#     def clean_plot(self, row):
#         x = row[0]
#         x = x.replace(np.nan,'unknown')
            
#     def transform(self, x, y=None):
#         final = np.array([self.clean_country(row) for row in x])
#         return final

#     def fit(self, x, y=None):
#         return self
>>>>>>> 383e56073a0b07b1fa74c5b6804d45e0b86fa24f

class CleanReleasedEncoder(BaseEstimator, TransformerMixin):
    """ class clean released encoder """
    def __init__(self):
        pass
            
    def transform(self, x, y=None):
        final =np.array([re.findall("[a-zA-Z]+",row[0])for row in x])
        return final

    def fit(self, x, y=None):
        return self


class CleanRatedEncoder(BaseEstimator, TransformerMixin):
    """ class clean rated encoder """
    def __init__(self):
        pass
        
    def clean_rated(self, row):
<<<<<<< HEAD
        # Taking the first elemnt if the list
=======
        # group different Rated labels
>>>>>>> 383e56073a0b07b1fa74c5b6804d45e0b86fa24f
        x = row[0]
        # group ratings 
        kids = ['TV-G', 'TV-PG', 'Kid', 'TV-Y7', 'TV-Y7-FV', 'TV-Y', 'E']
        teens = ['TV-13', 'TV-14', 'PG-13', 'PG', 'M']
        over_17 = ['TV-MA', 'NC-17', 'R', '18 and over', 'Unrated', 'UNRATED']
<<<<<<< HEAD
=======
        # general = ['G', 'APPROVED', 'Passed', 'M/PG', 'Approved', 'GP', 'X']
>>>>>>> 383e56073a0b07b1fa74c5b6804d45e0b86fa24f

        if x in kids:
            return ['kids']
        if x in teens:
            return ['teens']
        if x in over_17:
            return ['over_17']
        return ['General']
               
    def transform(self, x, y=None):
        final = np.array([self.clean_rated(row) for row in x])
        return final

    def fit(self, x, y=None):
        return self


class CleanAgeEncoder(BaseEstimator, TransformerMixin):
    """ class clean age encoder """
    def __init__(self):
        pass
    
    def diff_dates(self, row):
        x = row[0]
        d1 = datetime.now()
        d2 = pd.to_datetime(x, format='%Y')
        return [round(abs(d2 - d1) / np.timedelta64(1, 'Y'))]
    
    def transform(self, x, y=None):
        final = np.array([self.diff_dates(row) for row in x])
        return final
    
    def fit(self, x, y=None):
        return self

<<<<<<< HEAD

## -----------------------------

=======
###--------------------------------------
>>>>>>> 383e56073a0b07b1fa74c5b6804d45e0b86fa24f
   
if __name__ == "__main__":
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    df = load_data(5)
    test_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Not Available")),
        ("Transformer", CleanLanguageEncoder())
        ])
    test_transform = ColumnTransformer([
        ("Test", test_pipe, ['Language'])
        ], remainder="drop")
    
    final_pipe = Pipeline([
        ("Transform", test_transform)])
    
    x = df.drop(columns= ["avg_review_score"])
    y = df.avg_review_score
    
    final_pipe.fit_transform(x)