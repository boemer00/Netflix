import os
import pandas as pd
import numpy as np
import re

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
        
        #Transform Released Dates to Released Month
        df["Released"] = df["Released"].fillna("Non Available")
        def remove_digit(x):
            return ''.join([i for i in x if not i.isdigit()]).strip()
        df["Released_month"] = df["Released"].apply(remove_digit)
        df = df.drop(columns ="Released")
        
        #Join values from Rated Column
        
        #Kid
        df["Rated"] = df[["Rated"]].replace("TV-G","Kids")
        df["Rated"] = df[["Rated"]].replace("TV-PG","Kids")
        df["Rated"] = df[["Rated"]].replace("Kid","Kids")
        df["Rated"] = df[["Rated"]].replace("TV-Y7","Kids")
        df["Rated"] = df[["Rated"]].replace("TV-Y7-FV","Kids")
        df["Rated"] = df[["Rated"]].replace("TV-Y","Kids")

        #17 and over 
        df["Rated"] = df[["Rated"]].replace("TV-14","Teens")
        df["Rated"] = df[["Rated"]].replace("PG-13","Teens")
        df["Rated"] = df[["Rated"]].replace("TV-MA","Adult")
        df["Rated"] = df[["Rated"]].replace("NC-17","Adult")
        df["Rated"] = df[["Rated"]].replace("R","Adult")
        df["Rated"] = df[["Rated"]].replace("18 and over","Adult")
        df["Rated"] = df[["Rated"]].replace("Adult","17 and over")

        #Not Rated
        df["Rated"] = df[["Rated"]].fillna("Not Rated")
        df["Rated"] = df[["Rated"]].replace("Unrated","Not Rated")
        df["Rated"] = df[["Rated"]].replace("NOT RATED","Not Rated")
        df["Rated"] = df[["Rated"]].replace("UNRATED","Not Rated")
        df["Rated"] = df[["Rated"]].replace("E","Not Rated")


        #Teens
        df["Rated"] = df[["Rated"]].replace("TV-13","Teens")
        df["Rated"] = df[["Rated"]].replace("PG","Teens")
        df["Rated"] = df[["Rated"]].replace("M","Teens")


        #General
        df["Rated"] = df[["Rated"]].replace("G","General")
        df["Rated"] = df[["Rated"]].replace("APPROVED","Approved")
        df["Rated"] = df[["Rated"]].replace("Passed","General")
        df["Rated"] = df[["Rated"]].replace("M/PG","General")
        df["Rated"] = df[["Rated"]].replace("Approved","General")
        df["Rated"] = df[["Rated"]].replace("GP","General")
        df["Rated"] = df[["Rated"]].replace("X","Adult Movies")
        
        
        
        return self
    
    
if __name__ == "__main__":
    print(Netflix().load_data().head())
    print(Netflix().data_wrangling().head())
    