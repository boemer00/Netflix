import pandas as pd

def data_wrangling(self):
    """ WORK IN PROGRESS cleaning irrelevant rows and columns """ 
    
    df = self.drop(columns=['title', 'year', 'Awards', 'Poster', 'Metascore','totalSeasons', 'imdbVotes',
                            'Response','Index_match', 'DVD', 'BoxOffice','Production', 'Internet Movie Database',
                            'Website', 'Metacritic', 'Ratings'], inplace=True)
    
    ## fill nan and' min', convert to int and replace zero for the mean
    df['Runtime'] = df['Runtime'].fillna(0)
    df['Runtime'] = df['Runtime'].apply(lambda x: int(str(x).replace(' min', '')))
    df['Runtime'] = df['Runtime'].replace(0, df['Runtime'].mean())
    
    ## fill nan and remove '%', convert to float and replace zero for the mean
    df['Rotten Tomatoes'] = df['Rotten Tomatoes'].fillna(0) 
    df['Rotten Tomatoes'] = df['Rotten Tomatoes'].apply(lambda x: float(str(x).replace('%', '')))
    df['Rotten Tomatoes'] = df['Rotten Tomatoes'].replace(0, df['Rotten Tomatoes'].mean())
    
    ## replace countries with most frequent ('USA')
    freq_country = df[['Country']].value_counts().reset_index()['Country'][0]
    df['Country'] = df['Country'].replace(0, freq_country).replace('United States', freq_country)

    return self.df

def data_scaling(self):
    """ WORK IN PROGRESS """
    pass