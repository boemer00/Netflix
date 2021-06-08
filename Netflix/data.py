import os
import pandas as pd

class Movies:

    def get_data(self):
        """ get_data read all txt file and store them in one file """
        root = os.path.dirname(os.path.dirname(__file__))
        if not os.path.isfile('../../raw_data/data.csv'): 
            data = open('../../raw_data/data.csv', mode='w')
            row = list()
            raw_datasets = ['combined_data_1.txt', 'combined_data_2.txt',
                            'combined_data_3.txt', 'combined_data_4.txt']
            path_list = []
            for raw_dataset in raw_datasets:
                file_path = os.path.join(root, 'raw_data', raw_dataset)
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
        df = pd.read_csv('data.csv', sep=',', names=['movie','user','rating','date'])
        df.date = pd.to_datetime(df.date)

        # ratings sorted by date
        df.sort_values(by='date', inplace=True)
    
        return df


if __name__ == "__main__":
    print(Movies().get_data())
    
