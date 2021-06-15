import pandas as pd


def load_data(n):
    df = pd.read_csv("raw_data/merged_movies_by_index.csv", nrows=n)
    
    return df

    
if __name__ == "__main__":
   print(load_data.head())
    