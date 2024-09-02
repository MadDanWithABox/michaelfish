import configparser

from loguru import logger
from core.datasets import IrisDatasetLoader, BostonDatasetLoader, LondonWeatherLoader
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

config = configparser.ConfigParser()
config.read('.config')
data_path = config['settings']['DATA_PATH']
verbose = int(config['settings']['VERBOSE'])

class IrisTrainerInstance:
    def __init__(self):
        pass

    def load_data(self):
        self.data = IrisDatasetLoader()

    def train(self):
        try:
            classifier = svm.SVC()
            logger.info("Fetching dataset")
            X, y = self.data.load_data()
            logger.info("Training SVC Model")
            classifier.fit(X, y)
        except Exception as e:
            raise (e)
        return classifier


class BostonHousePriceTrainerInstance:
    def __init__(self):
        self.dataloader = BostonDatasetLoader()

    def train(self):
        self.X, self.y = self.dataloader.load_data()
        model = LinearRegression()
        model.fit(self.X, self.y)
        return model

class LondonWeatherTrainerInstance:
    def __init__(self):
        self.dataloader = LondonWeatherLoader(path=data_path)

    def train(self, use_full_data=False):
        self.X_train, _, self.Y_train, _ = self.dataloader.load_data(full=use_full_data)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.Y_train) 
        return self.model
    
    def eval(self, use_full_data=False):
        self.X_train, self.X, self.Y_train, self.Y = self.dataloader.load_data(full=use_full_data)
        predictions = self.model.predict(self.X)
                # Calculate evaluation metrics
        mae = mean_absolute_error(self.Y, predictions)
        mse = mean_squared_error(self.Y, predictions)
        r2 = r2_score(self.Y, predictions)
        
        # Create a summary string with the metrics
        eval_summary = {
            "mae": mae,
            "mse": mse,
            "r2": r2,
        }
        
        return eval_summary

class LondonWeatherAdvancedTrainerInstance(LondonWeatherTrainerInstance):
    def __init__(self, data_path="data.csv"):
        # Call the parent class's initializer
        super().__init__()
    
    def train(self):
        # Override train to always use full data
        return super().train(use_full_data=True)
    
    def eval(self):
        # Override eval to always use full data
        return super().eval(use_full_data=True)

