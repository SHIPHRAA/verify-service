import random
from image.models.schema import (
    ImageEstimationRequestBody,
    PostImageFileDetailRequestBody,
)

def _get_request_body(check_target_file_id: str):
    is_image_has_face = random.choice([True, False])
    if is_image_has_face:
        bounding_boxes = [
            PostImageFileDetailRequestBody.BoundingBox(
                x_position=random.randint(10, 100),
                y_position=random.randint(10, 100),
                width=random.randint(10, 100),
                height=random.randint(10, 100),
                fake_percentage=random.uniform(0, 1.0),
            )
            for _ in range(2)
        ]

        return PostImageFileDetailRequestBody(
            check_target_file_id=check_target_file_id,
            bounding_boxes=bounding_boxes,
            fake_percentage=round(random.uniform(0.0, 1.0), 2),
            general_object_fake_percentage=None,
        )
    else:
        return PostImageFileDetailRequestBody(
            check_target_file_id=check_target_file_id,
            bounding_boxes=[],
            fake_percentage=None,
            general_object_fake_percentage=round(random.uniform(0.0, 1.0), 2),
        )

def image_estimation_pipeline(
    authentication: str,
    image_estimation_request_body: ImageEstimationRequestBody,
):
    return _get_request_body(
        check_target_file_id=image_estimation_request_body.check_target_file_id
    ).model_dump()
