from pydantic import BaseModel, Field


class LondonWeatherRequestModel(BaseModel):
    date: int = Field(example=20200101, description="Date for weather prediction as an 8-digit string, in YYYYMMDD format")

class LondonWeatherResponseModel(BaseModel):
    predictionId: str = "f75ef3b8-f414-422c-87b1-1e21e684661c"
    predictedWeather: float 

class LondonWeatherAdvancedRequestModel(BaseModel):
    date: int = Field(example=20220319, description="Date int for weather prediction, in YYYYMMDD format")
    cloud_cover: float =  Field(example=2.0, description="Percentage cloud cover on the given day, between 0.0 and 100.0")
    sunshine: float = Field(example=10.0, description="Percentage sunshine on the day, bewteen 0.0 and 100.0")
    global_radiation: float = Field(example=44.0, description="Global radiation levels recorded on that day")
    precipitation: float = Field(example=42.0, description="Percentage precipitation")
    pressure: float = Field(example=16.6, description="Atmospheric pressure, ranges between ~95000 and ~105000")
    snow_depth: float = Field(example=11.1, description="Snow depth in mm")