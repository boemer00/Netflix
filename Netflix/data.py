import os
import pandas as pd
import numpy as np

class Movies:

    ROOT = os.path.dirname(os.path.dirname(__file__))
    def get_data(self):
        """ get_data read all txt file and store them in one file """
        if not os.path.isfile(os.path.join(self.ROOT,'raw_data','data.csv')): 
            data = open(os.path.join(self.ROOT,'raw_data','data.csv'), mode='w')
            row = list()
            raw_datasets = ['combined_data_1.txt', 'combined_data_2.txt',
                            'combined_data_3.txt', 'combined_data_4.txt']
            path_list = []
            for raw_dataset in raw_datasets:
                file_path = os.path.join(self.ROOT, 'raw_data', raw_dataset)
                path_list.append(file_path)
            
            for file in path_list:
                    with open(file) as f:
                        for line in f:
                            del row[:]
                            line = line.strip()
                            if line.endswith(':'):
                                movid_id = line.replace(':', '')
                            else:
                                row = [rows for rows in line.split(',')]
                                row.insert(0, movid_id)
                                data.write(','.join(row))
                                data.write('\n')
            data.close()
        
            # creating new dataframe from data.csv
            df = pd.read_csv(os.path.join(self.ROOT, 'raw_data', 'data.csv'), sep=',', names=['movie_id','user','rating','date'])

            # groupby 
            df = df.groupby('movie_id')['rating'].agg(['mean','count']).reset_index()
            
            # merge with movie titles (see --> def movie_titles)
            titles = self.movie_titles().merge(df, on='movie_id')
            titles.to_csv(os.path.join(self.ROOT, 'raw_data', 'data.csv'), index=False)
        
        return pd.read_csv(os.path.join(self.ROOT, 'raw_data', 'data.csv'))


    ## -----------------------------
    
    def movie_titles(self):
        """ import movie titles and add to grouped data file """
        df_title = pd.read_csv(os.path.join(self.ROOT, 'raw_data','movie_titles.csv'),
                               header = None, names = ['movie_id', 'Year', 'Name', 'a', 'b' ,'c'])
        df = df_title.drop(columns=['a', 'b', 'c'])
        return df


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
    print(Movies().get_data().head())
    print(Movies().movie_titles().head())
    print(Movies().data_wrangling().head())
    