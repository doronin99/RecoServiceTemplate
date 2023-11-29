from typing import List

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel

from service.api.exceptions import ModelNotFoundError, UserNotFoundError
from service.log import app_logger
from models.loader import load_model


# Define a Pydantic model for the response
class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


# Create an APIRouter instance
router = APIRouter()


# Define a health check endpoint
@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    """Health check endpoint"""
    return "I am alive"


# Define a recommendation endpoint
@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    """Get recommendations for a user based on the specified model."""
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    k_recs = request.app.state.k_recs

    # Check if the user_id is within a valid range
    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    # Retrieve recommendations based on the specified model
    if model_name == "top":
        reco = list(range(k_recs))
    elif model_name == "userknn":
        userknn_model = load_model("models/userknn.dill")
        reco = userknn_model.predict_single(user_id, N_recs=k_recs)
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    # Return the recommendations
    return RecoResponse(user_id=user_id, items=reco)


# Function to add the defined views to the FastAPI app
def add_views(app: FastAPI) -> None:
    app.include_router(router)
