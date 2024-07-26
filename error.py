import os
import pandas as pd

def preprocess_data(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    
    data = pd.read_csv(filepath)
    # Continue with preprocessing...

# Example usage
try:
    data = preprocess_data('tripv2pub 5.csv')
except FileNotFoundError as e:
    print(e)
