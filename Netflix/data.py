import os
import pandas as pd
import numpy as np
import re


def load_data(n):
    df = pd.read_csv("raw_data/merged_movies_by_index.csv", nrows=n)
    
    return df
    

## -----------------------------

def clean_runtime(x):
    counter = 0
    if 'h' in x:
        counter += int(x[0]) * 60
        x = re.sub('.*h', '', x).strip()
    x = x.replace('min', '').replace(',','').strip()
    counter += int(x)
    return counter

## -----------------------------

def data_wrangling(df):
    """ cleaning irrelevant rows and columns """ 

    # drop irrelevant columns
    df = df.drop(columns=['title', 'year', 'Awards', 'Poster', 'Metascore', 'DVD',
                        'BoxOffice', 'Internet Movie Database','totalSeasons',
                        'imdbVotes','Website', 'Response', 'Production', 'Metacritic', 'Ratings'])

    ## fill nan and' min', convert to int and replace zero for the mean
    df['Runtime'] = df['Runtime'].fillna("0").apply(clean_runtime)
    
    
    ## fill nan and remove '%', convert to float and replace zero for the mean
    df['Rotten Tomatoes'] = df['Rotten Tomatoes'].fillna(0) 
    df['Rotten Tomatoes'] = df['Rotten Tomatoes'].apply(lambda x: float(str(x).replace('%', '')))
    df['Rotten Tomatoes'] = df['Rotten Tomatoes'].replace(0, df['Rotten Tomatoes'].mean())
    
    ## dropna rows
    df = df.dropna(subset = ['Actors', 'Director', 'Writer', 'Language'])
    
    ## replace with other frequent values
    freq_country = df[['Country']].value_counts().reset_index()['Country'][0]
    df['Country'] = df['Country'].replace(0, freq_country).replace('United States', freq_country)
    freq_genre = df['Genre'].mode()[0]
    df['Genre'] = df['Genre'].replace(np.nan, freq_genre)
    df['Plot'] = df['Plot'].replace(np.nan,'unknown')
    
    return df

    
if __name__ == "__main__":
   print(load_data.head())
    