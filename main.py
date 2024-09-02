from loguru import logger
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from apis.v1.iris import router as iris_ns
from apis.v1.boston import router as boston_ns
from apis.v1.london import router as london_ns

import configparser

config = configparser.ConfigParser()
config.read('.config')

# Initialize logging
logger.add("./logs/katana.log", rotation="500 MB")
logger.info("Initializing application : MichaelFish")

app = FastAPI(
    title="Michael Fish Weather Forecasting⚡",
    version=1.0,
    description="Only the most accurate of weather forecasts 👩‍💻",
)
logger.info("Adding Iris namespace route")
app.include_router(iris_ns)
logger.info("Adding Boston namespace route")
app.include_router(boston_ns)
logger.info("Adding London Weather namespace route")
app.include_router(london_ns)


@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")
