from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Create the main FastAPI application
app = FastAPI(
    title="Verify API",
    description="API for image verification and reverse image search",
    version="1.0.0",
)

# FIXED: Add CORS middleware with specific origins and credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://34.146.50.68:3000",
        "https://34.146.50.68:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://34.146.50.68",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

from image.reverse_image_search import app as reverse_image_app

from image.routes import image_analysis

# Import the Twitter router from the video module
from video.twitter_search import router as twitter_router

# Include all routes from the reverse image search app
for route in reverse_image_app.routes:
    app.routes.append(route)

app.include_router(image_analysis.router)

# Register the Twitter router
app.include_router(twitter_router)


# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Add root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Verify API"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5500,
        reload=True,
        log_level="info",
    )
