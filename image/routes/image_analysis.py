from fastapi import APIRouter, Header, status
from fastapi.responses import JSONResponse
from image.models.schema import ImageEstimationRequestBody

from configs.env_config import settings

if settings.IS_GPU:
    from image.services.image_estimation_pipeline_gpu import image_estimation_pipeline
else:
    from image.services.image_estimation_pipeline_nogpu import image_estimation_pipeline

router = APIRouter(prefix="/image-analysis", tags=["image media"])

@router.post("/", response_model=dict, summary="Image Analysis API")
async def analyze_image(
    image_estimation_request_body: ImageEstimationRequestBody,
    authentication: str = Header(default="", alias="Authentication"),
):
    result = image_estimation_pipeline(
        authentication=authentication,
        image_estimation_request_body=image_estimation_request_body,
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"result": result}
    )
