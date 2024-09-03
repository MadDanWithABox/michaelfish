import uuid
from loguru import logger
from fastapi.routing import APIRouter
from apis.models.base import TrainingStatusResponse, EvalStatusResponse
from apis.models.weather import LondonWeatherAdvancedRequestModel, LondonWeatherRequestModel, LondonWeatherResponseModel
from core.trainer import LondonWeatherTrainerInstance, LondonWeatherAdvancedTrainerInstance
import pandas as pd
from math import pi, sin, cos
from datetime import datetime

router = APIRouter(prefix="/london")
#Could Load trained model. Dummy model being trained on startup...
logger.info("Training/Loading london weather classification model")
trainer = LondonWeatherTrainerInstance()
london_model = trainer.train()
logger.info("Training completed")

logger.info("Training/Loading advanced london weather classification model")
adv_trainer = LondonWeatherAdvancedTrainerInstance()
advanced_london_model = adv_trainer.train()
logger.info("Training completed")


@router.post(
    "/trainModel", tags=["london"], response_model=TrainingStatusResponse
)
async def london_train():
    training_id = uuid.uuid1()
    # Queue training / start training via RabbitMQ, Queue, etc..
    # Add task here
    # Track the id in a database
    return {
        "trainingId": str(training_id),
        "status": "Training started",
    }

@router.post(
    "/predictWeather", tags=["london"], response_model=LondonWeatherResponseModel
)
async def london_weather_prediction(body: LondonWeatherRequestModel):
    request = body.dict()
    date_int = request['date']  # Extract the date string from the request
    date_str= str(date_int)
    # Convert to datetime
    date = pd.to_datetime(date_str, format='%Y%m%d')

    # Extract features, maybe is more meaning in this fornat (weather cycles according to season)
    year = date.year
    month = date.month
    day = date.day
    day_of_week = date.dayofweek
    day_of_year = date.dayofyear
            
    # Apply sine and cosine transformations to encode seasonal patterns
    sin_doy = sin(2 * pi * day_of_year / 365)
    cos_doy = cos(2 * pi * day_of_year / 365)

    # Prepare payload
    payload = pd.DataFrame({
        'date': date_int,
        'year': [year],
        'month': [month],
        'day': [day],
        'day_of_week': [day_of_week],
        'sin_day_of_year': [sin_doy],
        'cos_doy': [cos_doy]
    })

    # Predict using the model
    prediction = london_model.predict(payload)
    
    # Return the result
    result = {
        "predictionId": str(uuid.uuid1()),
        "predictedWeather": prediction[0]
    }
    logger.info(result)
    return result

@router.post(
    "/evaluateWeather", tags = ["london"], response_model=EvalStatusResponse
)
async def london_model_evaluation():
    eval_id = uuid.uuid1()
    results = trainer.eval()
    return {
        "eval_id":str(uuid.uuid1()),
        "mean_absolute_error":results["mae"],
        "mean_squared_error":results["mse"],
        "r2_score":results["r2"]
        }


@router.post(
    "/trainModelAdvanced", tags=["london_advanced"], response_model=TrainingStatusResponse
)
async def london_train_advanced():
    training_id = uuid.uuid1()
    # Queue training / start training via RabbitMQ, Queue, etc..
    # Add task here
    # Track the id in a database
    return {
        "trainingId": str(training_id),
        "status": "Training started",
    }


@router.post(
    "/predictWeatherAdvanced", tags=["london_advanced"], response_model = LondonWeatherResponseModel
)
async def london_weather_prediction_advanced(body: LondonWeatherAdvancedRequestModel):
    request = body.dict()

    # Prepare the payload using all available features from the request
    payload = pd.DataFrame({
        'date': [request['date']],  # assuming date is in the format YYYYMMDD
        'cloud_cover': [request['cloud_cover']],
        'sunshine': [request['sunshine']],
        'global_radiation': [request['global_radiation']],
        'precipitation': [request['precipitation']],
        'pressure': [request['pressure']],
        'snow_depth': [request['snow_depth']]
    })

    # Predict using the advanced model
    prediction = advanced_london_model.predict(payload)

    # Return the result
    result = {
        "predictionId": str(uuid.uuid1()),
        "predictedWeather": prediction[0]
    }
    logger.info(result)
    return result



    
