import os
import pandas as pd
import numpy as np

class DataSourcing:

    
    def load_data(self):
        self.df = pd.read_csv("../raw_data/merged_movies_by_index.csv")
        
        return self
        


    ## -----------------------------
    
    def data_wrangling(self):
        """ cleaning irrelevant rows and columns """ 
    
        # drop irrelevant columns
        df = self.drop(columns=['title', 'year', 'Awards', 'Poster', 'Metascore', 'DVD',
                            'BoxOffice', 'Internet Movie Database','totalSeasons',
                            'imdbVotes','Website', 'Response', 'Production', 'Metacritic', 'Ratings'])

        ## fill nan and' min', convert to int and replace zero for the mean
        df['Runtime'] = df['Runtime'].fillna(0).apply(lambda x: str(x).replace(',', ''))
        df['Runtime'] = df['Runtime'].apply(lambda x: float(str(x).replace(' min', '')))
        df['Runtime'] = df['Runtime'].replace(0, df['Runtime'].mean())
        
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
        
        return self
    
    
if __name__ == "__main__":
    print(Netflix().load_data().head())
    print(Netflix().data_wrangling().head())
    