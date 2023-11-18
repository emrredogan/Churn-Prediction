import pandas as pd 

class DataLoader:
    def get_data(file_path):
        print(f"Data loaded from file path {file_path}")
        return pd.read_csv(file_path)