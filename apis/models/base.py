from pydantic import BaseModel


class TrainingStatusResponse(BaseModel):
    trainingId: str = "056b5d3d-f983-4cd3-8fbd-20b8dad24e0f"
    status: str = "Training  queued"


class EvalStatusResponse(BaseModel):
    eval_id: str
    mean_absolute_error: float
    mean_squared_error: float
    r2_score: float
