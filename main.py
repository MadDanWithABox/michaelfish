from loguru import logger
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from apis.v1.iris import router as iris_ns
from apis.v1.boston import router as boston_ns
from apis.v1.london import router as london_ns

import configparser
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client.exposition import start_http_server

config = configparser.ConfigParser()
config.read('.config')
verbose = int(config['settings']['VERBOSE'])

# Initialise Prometheus metrics
# Create Prometheus metrics
REQUEST_COUNT = Counter(
    "request_count", "Total number of requests", ["method", "endpoint", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Request latency in seconds", ["method", "endpoint"]
)

# Initialise logging
logger.add("./logs/katana.log", rotation="500 MB")
logger.info("Initializing application : MichaelFish")

app = FastAPI(
    title="Michael Fish Weather Forecasting⚡",
    version=1.0,
    description="Only the most accurate of weather forecasts 👩‍💻",
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    import time

    # Start the timer
    start_time = time.time()

    # Process the request
    response = await call_next(request)

    # Stop the timer and calculate the latency
    latency = time.time() - start_time

    # Update the metrics
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(latency)

    return response

if verbose == 1:
    logger.info("Adding Iris namespace route")
    logger.info("Adding Boston namespace route")
    logger.info("Adding London Weather namespace route")

app.include_router(iris_ns)
app.include_router(boston_ns)
app.include_router(london_ns)

# Metrics endpoint for Prometheus to scrape
@app.get("/metrics", include_in_schema=True)
async def metrics():
    return Response(generate_latest())


@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)