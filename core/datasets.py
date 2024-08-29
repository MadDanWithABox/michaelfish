from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from loguru import logger


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
        self.log_min_max_values()

    def log_min_max_values(self):
        logger.info("Logging min and max values for each column")
        for column in self.df.columns:
            min_value = self.df[column].min()
            max_value = self.df[column].max()
            #logger.info(f"Column: {column}, Min: {min_value}, Max: {max_value}")


    
    def load_data(self, full=False):
       
        # Handle missing values if any (for simplicity, we drop rows with missing values)
        self.df = self.df.dropna()

        # Split the data into features and target variable
        y = self.df['mean_temp']
        # we could use all the features to make a prediction, except for temperature stats
        if full:
            X = self.df.drop(columns=['mean_temp', 'min_temp', 'max_temp'])

        # or we could use the data as specified in the task, and do some feature engineering
        else:
            # Convert date to datetime format
            data_df = self.df
            data_df['date'] = pd.to_datetime(data_df['date'], format='%Y%m%d')
            # Extract date components
            data_df['year'] = data_df['date'].dt.year
            data_df['month'] = data_df['date'].dt.month
            data_df['day'] = data_df['date'].dt.day
            data_df['day_of_week'] = data_df['date'].dt.dayofweek
            data_df = data_df.drop(columns=['date','cloud_cover','sunshine','global_radiation','max_temp','min_temp','precipitation','pressure','snow_depth'])
            X = data_df
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # with more modern python, we could use mastch/case syntax here - but this is bound to 3.7 for module compat reasons
        if X_train.ndim == 1:
            logger.info("Dimensionality is not 2D. Reshaping...")
            X_train = X_train.values.reshape(-1, 1)
        if X_test.ndim == 1:
            logger.info("Dimensionality is not 2D. Reshaping...")
            X_test = X_test.values.reshape(-1, 1)
        return X_train, X_test, y_train, y_test