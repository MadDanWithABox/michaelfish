from math import pi, sin, cos
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from loguru import logger

import configparser

config = configparser.ConfigParser()
config.read('.config')

verbose = int(config['settings']['VERBOSE'])

class IrisDatasetLoader:
    def __init__(self):
        self.data = datasets.load_iris()

    def load_data(self):
        X = self.data.data
        y = self.data.target
        return X, y


class BostonDatasetLoader:
    def __init__(self):
        self.X = None
        self.y = None

    def load_data(self):
        self.X, self.y = datasets.load_boston(return_X_y=True)
        return self.X, self.y

class LondonWeatherLoader:
    def __init__(self, path):
        self.df = pd.read_csv(path)
    
    def load_data(self, full=False):
       
        # Handle missing values if any (for simplicity, we drop rows with missing values)
        self.df = self.df.dropna()

        # Split the data into features and target variable
        y = self.df['mean_temp']
        
        if full:
            # Drop mean_temp and temperature statistics if using full data
            X = self.df.drop(columns=['mean_temp', 'min_temp', 'max_temp'])
        else:
            # Convert date to datetime format
            data_df = self.df
            data_df['date'] = pd.to_datetime(data_df['date'], format='%Y%m%d')
            
            # Extract date components
            data_df['year'] = data_df['date'].dt.year
            data_df['month'] = data_df['date'].dt.month
            data_df['day'] = data_df['date'].dt.day
            data_df['day_of_week'] = data_df['date'].dt.dayofweek

            # Calculate day of the year
            data_df['day_of_year'] = data_df['date'].dt.dayofyear
            
            # Apply sine and cosine transformations to encode seasonal patterns
            data_df['sin_day_of_year'] = data_df['day_of_year'].apply(lambda x: sin(2 * pi * x / 365))
            data_df['cos_day_of_year'] = data_df['day_of_year'].apply(lambda x: cos(2 * pi * x / 365))
            
            # Drop unnecessary columns, including date and the original day_of_year
            data_df = data_df.drop(columns=[
                'date', 'day_of_year', 'cloud_cover', 'sunshine', 
                'global_radiation', 'max_temp', 'min_temp', 
                'precipitation', 'pressure', 'snow_depth'
            ])
            X = data_df
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Ensure correct dimensionality for the training and test data
        if X_train.ndim == 1:
            logger.info("Dimensionality is not 2D. Reshaping...")
            X_train = X_train.values.reshape(-1, 1)
        if X_test.ndim == 1:
            logger.info("Dimensionality is not 2D. Reshaping...")
            X_test = X_test.values.reshape(-1, 1)
        
        return X_train, X_test, y_train, y_test
