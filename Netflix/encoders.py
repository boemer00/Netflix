from os import X_OK
import re
from typing import final
import pandas as pd
import numpy as np
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
    # class clean runtime encoder(CustomEncoder):
    def __init__(self):
        pass

    def replace_strings(self, row): 
        """ regex and replace str formats """
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

## IGNORED THE BELOW AS THE ROTTEN TOMATOES COLUMN IS BEING DROPPED
# class CleanTomatoesEncoder(BaseEstimator, TransformerMixin):
#     # class clean rottenTomatoes encoder(CustomEncoder):
#     def __init__(self):
#         pass
        
#     def clean_tomatoes(self, row):
#         """ fill nan and remove '%', convert to float and replace zero for the mean """
#         x = row[0]
#         print(row)
#         x = x.replace('%', '')
#        # x = x.replace(0, x.mean())
#         return X
    
#     def transform(self, x, y=None):
#         """ fill nan and remove '%', convert to float and replace zero for the mean """
#         final = np.array([self.clean_tomatoes(row) for row in x])
#         return final

#     def fit(self, x, y=None):
#         return self

class CleanLanguageEncoder(BaseEstimator, TransformerMixin):
    # class clean language encoder(CustomEncoder):
    def __init__(self):
        pass
        
    def include_english(self, row):
        """ replace with other frequent values """
        x = row[0]
        if "english" in x.lower():
            return [1]
        return [0]
            
    def transform(self, x, y=None):
        final = np.array([self.include_english(row) for row in x])
        print(final)
        return final
        
    def fit(self, x, y=None):
        return self

class CleanCountryEncoder(BaseEstimator, TransformerMixin):
    # class clean country encoder(CustomEncoder):
    def __init__(self):
        pass
        
    def include_us(self, row):
        """ replace with other frequent values """
        x = row[0]
        usa = ["United States", "USA"]
        for name in usa:
            if name in x:
                return [1]
        return [0]
            
    def transform(self, x, y=None):
        final = np.array([self.include_us(row) for row in x])
        return final
        
    def fit(self, x, y=None):
        return self

# # Ignore below as already included in imputer
# class CleanGenreEncoder(BaseEstimator, TransformerMixin):
#     # class clean genre encoder(CustomEncoder):
#     def __init__(self):
#         pass
        
#     def clean_genre(self, row):
#         """ replace with other frequent values """
#         x = row[0]
#         freq_genre = x.mode()[0]
#         x = x.replace(np.nan, freq_genre)  
#         return x 
            
#     def transform(self, x, y=None):        
#         final = np.array([self.clean_genre(row) for row in x])
#         return final

#     def fit(self, x, y=None):
#         return self
    
# #ignore for now as NLP not confirmed
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

class CleanReleasedEncoder(BaseEstimator, TransformerMixin):
    # class clean plot encoder(CustomEncoder):
    def __init__(self):
        pass
            
    def transform(self, x, y=None):
        final =np.array([re.findall("[a-zA-Z]+",row[0])for row in x])
        return final

    def fit(self, x, y=None):
        return self

class CleanRatedEncoder(BaseEstimator, TransformerMixin):
    # class clean plot encoder(CustomEncoder):
    def __init__(self):
        pass
        
    def clean_rated(self, row):
        """ Group different Rated labels """
        x = row[0]
        # group ratings 
        kids = ['TV-G', 'TV-PG', 'Kid', 'TV-Y7', 'TV-Y7-FV', 'TV-Y', 'E']
        teens = ['TV-13', 'TV-14', 'PG-13', 'PG', 'M']
        over_17 = ['TV-MA', 'NC-17', 'R', '18 and over', 'Unrated', 'UNRATED']
        #general = ['G', 'APPROVED', 'Passed', 'M/PG', 'Approved', 'GP', 'X']

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

##to do class CleanAgeEncoder


## -----------------------------
## -----------------------------
## -----------------------------

# def load_data(n):
#     """ upload the df_train.csv """
#     df = pd.read_csv('raw_data/df_train.csv', nrows=n)
#     return df
    
## -----------------------------

# def drop_columns(df):
#     """ drop irrelevant columns and rows """
#     df = df.drop(columns=['title', 'year', 'Awards', 'Poster',
#                           'Metascore', 'DVD', 'BoxOffice',
#                           'Internet Movie Database', 'totalSeasons', 
#                           'imdbVotes','Website', 'Response',
#                           'Production', 'Metacritic', 'Ratings'])
#     df = df.dropna(subset = ['Actors', 'Director', 'Writer', 'Language'])
    

## -----------------------------

# def clean_runtime(df):
#     """ regex and replace str formats """
#     x = df['Runtime']
#     counter = 0
#     if 'h' in x:
#         counter += int(x[0]) * 60
#         x = re.sub('.*h', '', x).strip()
#     x = x.replace('min', '').replace(',','').strip()
#     counter += int(x)
#     return counter

# def apply_runtime(df):
#     """ fill nan and' min', convert to int and replace zero for the mean """
#     df['Runtime'] = df['Runtime'].fillna("0").apply(clean_runtime)
    

## -----------------------------

# def clean_tomatoes(df):
#     """ fill nan and remove '%', convert to float and replace zero for the mean """
#     df['Rotten Tomatoes'] = df['Rotten Tomatoes'].fillna(0) 
#     df['Rotten Tomatoes'] = df['Rotten Tomatoes'].apply(lambda x: float(str(x).replace('%', '')))
#     df['Rotten Tomatoes'] = df['Rotten Tomatoes'].replace(0, df['Rotten Tomatoes'].mean())
    

## -----------------------------

# def clean_country_genre_plot(df):
#     """ replace with other frequent values """
#     freq_country = df[['Country']].value_counts().reset_index()['Country'][0]
#     df['Country'] = df['Country'].replace(0, freq_country).replace('United States', freq_country)
#     freq_genre = df['Genre'].mode()[0]
#     df['Genre'] = df['Genre'].replace(np.nan, freq_genre)
#     df['Plot'] = df['Plot'].replace(np.nan,'unknown')
    

## -----------------------------

# def clean_released(df):
#     """ transform Released Date to Released Month """
#     df['Released'] = df['Released'].fillna('Non Available')
    
#     def remove_digit(x):
#         return ''.join([i for i in x if not i.isdigit()]).strip("-")
    
#     df['Released_month'] = df['Released'].apply(remove_digit)
#     df = df.drop(columns ='Released')
    

## -----------------------------

# def clean_rated(df):
#     """ Group different Rated labels """
#     # fill na
#     df['Rated'] = df[['Rated']].fillna("Not Rated")

#     # group ratings 
#     kids = ['TV-G', 'TV-PG', 'Kid', 'TV-Y7', 'TV-Y7-FV', 'TV-Y']
#     teens = ['TV-13', 'TV-14', 'PG-13', 'PG', 'M']
#     over_17 = ['TV-MA', 'NC-17', 'R', '18 and over']
#     not_rated = ['Unrated', 'NOT RATED', 'UNRATED', 'E']
#     general = ['G', 'APPROVED', 'Passed', 'M/PG', 'Approved', 'GP', 'X']

#     # replace ratings
#     df['Rated'] = df['Rated'].replace(kids, 'Kids')\
#                              .replace(teens, 'Teens')\
#                              .replace(over_17, 'Above 17')\
#                              .replace(not_rated, 'Not Rated')\
#                              .replace(general, 'General')

## -----------------------------
 

   
if __name__ == "__main__":
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
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