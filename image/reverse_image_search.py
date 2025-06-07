from fastapi import FastAPI, HTTPException, status, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import json
import base64
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union, Tuple
from datetime import datetime
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import tempfile
import os
import time
import logging
import uuid
import uvicorn
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import socket
import sys
from sqlalchemy.sql import text
from sqlalchemy import (
    Boolean,
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.exc import OperationalError
from sqlalchemy.engine.url import URL
from .websearch import process_url
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

load_dotenv()


# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("image_search_api")

# Database configuration
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "34.84.7.179"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "database": os.environ.get("DB_NAME", "focust_db"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "3@SfnaVu40aO%Ryf"),
    "sslmode": "require",
}

DATABASE_URL = URL.create(
    "postgresql",
    username=DB_CONFIG["user"],
    password=DB_CONFIG["password"],
    host=DB_CONFIG["host"],
    port=DB_CONFIG["port"],
    database=DB_CONFIG["database"],
    query={"sslmode": "require"},
)


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


# Function to test database connectivity before creating engine
def test_db_connection():
    try:
        logger.info(
            f"Testing connection to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}..."
        )
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((DB_CONFIG["host"], int(DB_CONFIG["port"])))
        sock.close()

        if result != 0:
            logger.error(
                f"Cannot connect to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}"
            )
            logger.error(
                f"Check if the database server is running and accessible from this network"
            )
            logger.error(
                f"Check if there are any firewall rules blocking the connection"
            )
            logger.error(
                f"Database connection is required for this application to work"
            )
            sys.exit(1)  # Exit the application if database connection fails
        else:
            logger.info(
                f"Successfully connected to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}"
            )

    except Exception as e:
        logger.error(f"Error testing database connection: {str(e)}")
        logger.error(f"Database connection is required for this application to work")
        sys.exit(1)  # Exit the application if database connection fails


# Test database connectivity before proceeding
test_db_connection()

# Create SQLAlchemy engine
try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    # Try a quick connection
    with engine.connect() as conn:
        logger.info("Successfully connected to the database.")

except OperationalError as e:
    logger.error(f"Database connection error: {str(e)}")
    logger.error("Please check your database credentials and network connectivity.")
    logger.error("Database connection is required for this application to work.")
    sys.exit(1)  # Exit the application if database connection fails

except Exception as e:
    logger.error(f"Unexpected database error: {str(e)}")
    logger.error("Database connection is required for this application to work.")
    sys.exit(1)  # Exit the application if database connection fails


# Define the request model
class ImageSearchRequest(BaseModel):
    image_url: str
    tweet_id: str  # Required field


# Define the response models
class CheckTargetImageRelatedContentDomain(BaseModel):
    id: Optional[int] = None
    company_id: Optional[int] = None
    check_target_file_id: Optional[int] = None
    image_path: str
    summary: Optional[str] = None
    related_percentage: Optional[float] = None
    url: str
    url_page_title: Optional[str] = None
    url_page_date: Optional[datetime] = None

    # Fields for enhanced contenfrom websearch import is_twitter_url, extract_tweet_id, get_tweet_by_id, format_tweet_info, extract_article_info, process_url
    content_title: Optional[str] = None
    content_date: Optional[str] = None
    content_publisher: Optional[str] = None
    content_summary: Optional[str] = None
    content_processed: Optional[bool] = False

    # Add a method to convert to dictionary for JSON serialization
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "image_path": self.image_path,
            "url": self.url,
        }

        if self.id is not None:
            result["id"] = self.id
        if self.company_id is not None:
            result["company_id"] = self.company_id
        if self.check_target_file_id is not None:
            result["check_target_file_id"] = self.check_target_file_id
        if self.summary is not None:
            result["summary"] = self.summary
        if self.related_percentage is not None:
            result["related_percentage"] = self.related_percentage
        if self.url_page_title is not None:
            result["url_page_title"] = self.url_page_title
        if self.url_page_date is not None:
            result["url_page_date"] = self.url_page_date.isoformat()

        # Include enhanced content if processed
        if self.content_processed:
            result["content"] = {
                "title": self.content_title,
                "date": self.content_date,
                "publisher": self.content_publisher,
                "summary": self.content_summary,
            }

        return result


class ImageSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    count: int
    search_time: float
    query_id: str


class TranslateRequest(BaseModel):
    results: List[Dict[str, Any]]
    target_language: str
    target_language_name: Optional[str] = None


class TranslateResponse(BaseModel):
    translated_results: List[Dict[str, Any]]
    original_count: int
    translated_count: int
    target_language: str
    target_language_name: str


class BusinessException(Exception):
    status_code: int
    content: str

    def __init__(self, status_code: int, content: str):
        self.status_code = status_code
        self.content = content


# Database Models
class ImageSearchQuery(Base):
    __tablename__ = "image_search_queries"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(String, unique=True, index=True)
    image_url = Column(String)
    tweet_id = Column(String, index=True)  # Required field
    search_time = Column(Float)
    created_at = Column(DateTime, default=datetime.now)

    # Relationship to results
    results = relationship("ImageSearchResult", back_populates="query")


class ImageSearchResult(Base):
    __tablename__ = "image_search_results"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(String, ForeignKey("image_search_queries.query_id"))
    image_path = Column(String)
    url = Column(String)
    related_percentage = Column(Float, nullable=True)
    url_page_title = Column(String, nullable=True)
    url_page_date = Column(DateTime, nullable=True)
    summary = Column(Text, nullable=True)

    # Added columns for enhanced contenfrom websearch import is_twitter_url, extract_tweet_id, get_tweet_by_id, format_tweet_info, extract_article_info, process_url
    content_title = Column(String, nullable=True)
    content_date = Column(String, nullable=True)
    content_publisher = Column(String, nullable=True)
    content_summary = Column(Text, nullable=True)
    content_processed = Column(Boolean, default=False)
    content_processed_at = Column(DateTime, nullable=True)

    # Relationship to query
    query = relationship("ImageSearchQuery", back_populates="results")


# Constants
TIME_OUT_SETTING = (30.0, 30.0)
GOOGLE_CLOUD_VISION_API_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
RESULTS_INT = 100

SUPPORTED_LANGUAGES_MAP = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ar": "Arabic",
    "hi": "Hindi",
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "tr": "Turkish",
}


# Create FastAPI app
app = FastAPI(
    title="Image Search API",
    description="API for searching similar images on the web using Google Cloud Vision API",
    version="1.0.0",
    default_response_class=UTF8JSONResponse,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize database tables on startup
@app.on_event("startup")
def startup_event():
    try:
        # Create tables if they don't exist
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {str(e)}")
        logger.error("Database initialization is required for this application to work")
        sys.exit(1)  # Exit the application if database initialization fails


# Functions
def get_access_token() -> str:
    """
    Get an access token for the Google Cloud Vision API using a service account key file.

    Returns:
        str: The access token.
    """
    try:
        credentials_path = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS", "credentials.json"
        )

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=["https://www.googleapis.com/auth/cloud-vision"]
        )

        request = Request()
        credentials.refresh(request)

        return credentials.token
    except Exception as e:
        logger.error(f"Failed to get access token: {str(e)}")
        raise e


def download_image_from_url(
    image_url: str,
) -> Tuple[Optional[str], Optional[Exception]]:
    """
    Download an image from a URL and save it to a temporary file.

    Args:
        image_url: URL of the image to download

    Returns:
        Tuple containing either the path to the temporary file or None, and an Exception or None
    """
    try:
        logger.debug(f"Downloading image from URL: {image_url}")

        # Create a temporary file
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp_file_path = tmp_file.name
        tmp_file.close()

        # Download the image
        response = requests.get(
            image_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/97.0.4692.99 Safari/537.36"
                ),
            },
            timeout=TIME_OUT_SETTING,
        )

        if response.status_code != 200:
            os.unlink(tmp_file_path)
            logger.error(
                f"Failed to download image, status code: {response.status_code}"
            )
            return None, HTTPException(
                status_code=response.status_code,
                detail=f"Failed to download image from URL: {image_url}",
            )

        # Save the image to the temporary file
        with open(tmp_file_path, "wb") as f:
            f.write(response.content)

        return tmp_file_path, None

    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        # Clean up the temporary file if it was created
        if "tmp_file_path" in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        return None, e


def request(
    path, method, headers={}, data={}, params={}, files={}
) -> Tuple[Optional[requests.Response], Optional[Union[BusinessException, Exception]]]:
    """
    Send a request to an external API.

    Args:
        path: URL path
        method: HTTP method
        headers: Request headers
        data: Request data
        params: URL parameters
        files: Files to upload

    Returns:
        Tuple containing either a response or a BusinessException
    """
    # Check method
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    if method not in methods:
        return None, BusinessException(
            status_code=status.HTTP_400_BAD_REQUEST, content=f"Invalid method {method}"
        )

    # Access
    res, err = _request(path, method, headers, data, params, files)
    if err:
        return None, err

    # Handle non-200 status codes
    if res.status_code != 200:
        res_content = res.text.replace("\n", "\\n")
        logger.error(f"External API returned non-200 status code: {res.status_code}")
        return None, BusinessException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=f"External API returned status code {res.status_code}",
        )

    return res, err


def _request(
    url, method, headers, data, params, files
) -> Tuple[Optional[requests.Response], Optional[Exception]]:
    """
    Internal function to make HTTP requests.
    """
    try:
        if method == "GET":
            res = requests.get(
                url, headers=headers, params=params, timeout=TIME_OUT_SETTING
            )
        elif method == "POST":
            if headers.get("Content-Type") == "multipart/form-data":
                res = requests.post(
                    url, headers=headers, files=files, timeout=TIME_OUT_SETTING
                )
            else:
                res = requests.post(
                    url, headers=headers, json=data, timeout=TIME_OUT_SETTING
                )
        elif method == "PUT":
            res = requests.put(
                url, headers=headers, json=data, timeout=TIME_OUT_SETTING
            )
        elif method == "PATCH":
            res = requests.patch(
                url, headers=headers, json=data, timeout=TIME_OUT_SETTING
            )
        elif method == "DELETE":
            res = requests.delete(
                url, headers=headers, json=data, timeout=TIME_OUT_SETTING
            )
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        return None, BusinessException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=f"Request error: {str(e)}",
        )
    return res, None


def _res_2_domains(
    res_body: str, original_image_url: str = None
) -> List[CheckTargetImageRelatedContentDomain]:
    """
    Extract domain information from Google Cloud Vision API response.
    """
    domains = []
    res_body_dict = json.loads(res_body)

    if "responses" in res_body_dict and len(res_body_dict["responses"]) > 0:
        response = res_body_dict["responses"][0]
        if "webDetection" in response:
            web_detection = response["webDetection"]

            # Extract both full and partial matches
            matches_count = 0
            if "pagesWithMatchingImages" in web_detection:
                matches_count = len(web_detection["pagesWithMatchingImages"])
                logger.info(f"Found {matches_count} pages with matching images")

                for pages_with_matching_image in web_detection[
                    "pagesWithMatchingImages"
                ]:
                    try:
                        if "url" in pages_with_matching_image:
                            url = pages_with_matching_image["url"]

                            # Get the matching image URL
                            image_path = None
                            if (
                                "partialMatchingImages" in pages_with_matching_image
                                and len(
                                    pages_with_matching_image["partialMatchingImages"]
                                )
                                > 0
                            ):
                                if (
                                    "url"
                                    in pages_with_matching_image[
                                        "partialMatchingImages"
                                    ][0]
                                ):
                                    image_path = pages_with_matching_image[
                                        "partialMatchingImages"
                                    ][0]["url"]
                            elif (
                                "fullMatchingImages" in pages_with_matching_image
                                and len(pages_with_matching_image["fullMatchingImages"])
                                > 0
                            ):
                                if (
                                    "url"
                                    in pages_with_matching_image["fullMatchingImages"][
                                        0
                                    ]
                                ):
                                    image_path = pages_with_matching_image[
                                        "fullMatchingImages"
                                    ][0]["url"]

                            # Use original image if no matching image was found
                            if not image_path and original_image_url:
                                image_path = original_image_url

                            # Calculate a score (higher for full matches)
                            score = None
                            if "fullMatchingImages" in pages_with_matching_image:
                                score = 99.0
                            elif "partialMatchingImages" in pages_with_matching_image:
                                score = 95.0

                            if url and image_path:
                                domains.append(
                                    CheckTargetImageRelatedContentDomain(
                                        image_path=image_path,
                                        url=url,
                                        related_percentage=score,
                                    )
                                )
                    except Exception as e:
                        logger.error(f"Error processing match: {str(e)}")
                        continue

    # If no domains were found and we have the original image, return one result with the original image
    if len(domains) == 0 and original_image_url:
        logger.info("No matching images found, using original image as fallback")
        domains.append(
            CheckTargetImageRelatedContentDomain(
                image_path=original_image_url,
                url="https://www.tiktok.com",  # Default URL
                related_percentage=99.0,
                url_page_title="No title available",
            )
        )

    return domains


def request_google_cloud_vision_api(image_data, original_image_url: str = None):
    """
    Makes a request to Google Cloud Vision API to find web matches for an image.
    """
    try:
        # Check if image_data is a file path or binary data
        if isinstance(image_data, str) and os.path.isfile(image_data):
            with open(image_data, "rb") as image:
                image_content = image.read()
        else:
            image_content = image_data

        # Get authentication token
        access_token = get_access_token()

        # Set up API request headers
        headers = {
            "Authorization": f"Bearer {access_token}",
            "x-goog-user-project": "focust-dev",
            "Content-Type": "application/json; charset=utf-8",
        }

        # Prepare request data with base64 encoded image
        data = {
            "requests": [
                {
                    "image": {
                        "content": base64.b64encode(image_content).decode("utf-8")
                    },
                    "features": [{"maxResults": RESULTS_INT, "type": "WEB_DETECTION"}],
                }
            ]
        }

        logger.info("Sending request to Google Cloud Vision API")
        # Make API request
        res, err = request(
            path=GOOGLE_CLOUD_VISION_API_ENDPOINT,
            method="POST",
            headers=headers,
            data=data,
        )

        # Handle request error
        if err is not None:
            logger.error(f"Google Cloud Vision API error: {str(err)}")
            return None, err

        # Process the response
        logger.info("Processing Google Cloud Vision API response")
        return _res_2_domains(res.text, original_image_url), None

    except Exception as e:
        logger.error(f"Google Cloud Vision API request error: {str(e)}")
        return None, e


def _get_url_page_title_and_time(
    url: str,
) -> Tuple[Optional[str], Optional[datetime], Optional[BaseException]]:
    """
    Fetch the title and publication date from a web page.
    """
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            url = f"http://{url}"

        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/97.0.4692.99 Safari/537.36"
                ),
            },
            timeout=TIME_OUT_SETTING,
        )

        if response.status_code != 200:
            logger.warning(
                f"Non-200 response when fetching page metadata: {url}, status: {response.status_code}"
            )
            return (
                None,
                None,
                BusinessException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content=f"Web page request returned non-200 response | url: {url} | status_code: {response.status_code}",
                ),
            )

        soup = BeautifulSoup(response.content, "html.parser")

        # Get page title
        title_tag = soup.find("title")
        url_page_title = title_tag.get_text() if title_tag else None

        # Try multiple methods to find the date
        url_page_date = None

        # Method 1: Look for time tag with datetime attribute
        time_tag = soup.find("time")
        if time_tag and time_tag.has_attr("datetime"):
            try:
                url_page_date = datetime.fromisoformat(
                    time_tag["datetime"].replace("Z", "+00:00")
                )
                return url_page_title, url_page_date, None
            except Exception:
                pass

        # Method 2: Look for meta tags with publication date information
        date_meta_tags = [
            soup.find("meta", attrs={"property": "article:published_time"}),
            soup.find("meta", attrs={"property": "og:published_time"}),
            soup.find("meta", attrs={"itemprop": "datePublished"}),
            soup.find("meta", attrs={"name": "date"}),
            soup.find("meta", attrs={"name": "pubdate"}),
            soup.find("meta", attrs={"name": "publication_date"}),
            soup.find("meta", attrs={"name": "publish_date"}),
        ]

        for meta_tag in date_meta_tags:
            if meta_tag and meta_tag.has_attr("content"):
                try:
                    date_str = meta_tag["content"]
                    try:
                        # ISO format dates
                        url_page_date = datetime.fromisoformat(
                            date_str.replace("Z", "+00:00")
                        )
                        break
                    except ValueError:
                        try:
                            # RFC 822 format dates
                            from email.utils import parsedate_to_datetime

                            url_page_date = parsedate_to_datetime(date_str)
                            break
                        except (ValueError, TypeError):
                            pass
                except Exception:
                    pass

        return url_page_title, url_page_date, None
    except Exception as e:
        import traceback

        logger.error(f"Error fetching page metadata: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return (
            None,
            None,
            BusinessException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=f"Failed to fetch web page | url: {url}",
            ),
        )


# Image similarity calculation functions
def _load_and_resize_image(tmp_file_path: str, image: np.ndarray):
    """Load image from file and resize to match the reference image."""
    tmp_image = cv2.imread(tmp_file_path)
    if tmp_image is None:
        raise ValueError(f"Failed to load image: {tmp_file_path}")
    return cv2.resize(tmp_image, (image.shape[1], image.shape[0]))


def _calculate_ssim_similarity(tmp_file_path: str, image: np.ndarray) -> float:
    """Calculate SSIM (Structural Similarity Index) between two images."""
    try:
        tmp_image = _load_and_resize_image(tmp_file_path, image)

        # Convert images to grayscale
        tmp_image_gray = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        similarity, _ = ssim(image_gray, tmp_image_gray, full=True)
        return similarity
    except Exception as e:
        logger.error(f"SSIM calculation error: {str(e)}")
        return None


def _calculate_related_percentage(
    tmp_file_path: str, image: np.ndarray, algorithm: str = "ssim"
) -> float:
    """Calculate image similarity using the specified algorithm."""
    # Currently only implementing SSIM as it's the most reliable
    return _calculate_ssim_similarity(tmp_file_path=tmp_file_path, image=image)


def update_domain_with_metadata(
    domains: List[CheckTargetImageRelatedContentDomain], limit: int = 100
) -> List[CheckTargetImageRelatedContentDomain]:
    """
    Update domain list with web page titles and dates.
    """
    # Apply limit
    domains = domains[:limit]

    total = len(domains)
    logger.info(f"Updating metadata for {total} domains")

    for i, domain in enumerate(domains):
        title, date, err = _get_url_page_title_and_time(domain.url)
        if not err:
            domain.url_page_title = title or "No title available"
            domain.url_page_date = date
        else:
            # Still set a title if available
            if domain.url_page_title is None:
                parsed_url = urlparse(domain.url)
                domain.url_page_title = f"Content from {parsed_url.netloc}"

        # Add a small delay to avoid overwhelming servers
        if i % 5 == 0 and i > 0:
            time.sleep(1)

    return domains


# Process URL contentfrom websearch import is_twitter_url, extract_tweet_id, get_tweet_by_id, format_tweet_info, extract_article_info, process_url
def process_and_update_url_content(
    db: Session, result_id: int, url: str, tweet_id: str
) -> bool:
    """
    Process a URL to extract article or tweet information and update the search result.

    Args:
        db: Database session
        result_id: ID of the search result to update
        url: URL to process
        tweet_id: Tweet ID associated with the search

    Returns:
        Boolean indicating success
    """
    try:
        # Get the search result
        result = (
            db.query(ImageSearchResult)
            .filter(ImageSearchResult.id == result_id)
            .first()
        )
        if not result:
            logger.error(f"Search result not found: {result_id}")
            return False

        # Get OpenAI API key from environment variable
        api_key = os.environ.get("OPENAI_API_KEY")

        # Process the URL usifrom websearch import is_twitter_url, extract_tweet_id, get_tweet_by_id, format_tweet_info, extract_article_info, process_url module
        logger.info(f"Processing URL: {url} for result ID: {result_id}")
        content_data = process_url(url, api_key)

        if content_data:
            # Update the search result with content information
            result.content_title = content_data.get("title")
            result.content_date = content_data.get("date")
            result.content_publisher = content_data.get("publisher")
            result.content_summary = content_data.get("summary")
            result.content_processed = True
            result.content_processed_at = datetime.now()

            # Only update the summary if it doesn't already exist
            if not result.summary and content_data.get("summary"):
                result.summary = content_data.get("summary")

            # Commit the changes
            db.commit()
            logger.info(f"Updated content info for result ID: {result_id}, URL: {url}")
            return True
        else:
            logger.warning(f"No content data returned for URL: {url}")
            return False

    except Exception as e:
        db.rollback()
        logger.error(f"Error processing URL content: {str(e)}")
        return False


# Process all URLs for a query
def process_all_urls_for_query(query_id: str, tweet_id: str):
    """Process all URLs for a specific query ID in the background."""
    try:
        # Create a new database session for this background task
        db = SessionLocal()

        try:
            # Get all results for this query
            results = (
                db.query(ImageSearchResult)
                .filter(ImageSearchResult.query_id == query_id)
                .all()
            )
            logger.info(
                f"Processing {len(results)} URLs for query {query_id} in background"
            )

            # Process each result URL
            for result in results:
                process_and_update_url_content(db, result.id, result.url, tweet_id)

            logger.info(f"Completed background processing for query {query_id}")
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error in background URL processing: {str(e)}")


# Function to save search results to database
def save_search_results_to_db(
    db: Session,
    image_url: str,
    tweet_id: str,
    search_time: float,
    results: List[Dict[str, Any]],
) -> str:
    """
    Save search results to the database with batching to avoid oversized queries.
    """
    try:
        # Generate a unique query ID
        query_id = str(uuid.uuid4())

        # Create query record
        query = ImageSearchQuery(
            query_id=query_id,
            image_url=image_url,
            tweet_id=tweet_id,
            search_time=search_time,
        )
        db.add(query)
        db.commit()  # Commit the query first to ensure it exists

        # Create result records in smaller batches to avoid oversized queries
        BATCH_SIZE = 20  # Process 20 results at a time

        for i in range(0, len(results), BATCH_SIZE):
            batch = results[i : i + BATCH_SIZE]

            # Add each result in the batch
            for result in batch:
                db_result = ImageSearchResult(
                    query_id=query_id,
                    image_path=result.get("image_path"),
                    url=result.get("url"),
                    related_percentage=result.get("related_percentage"),
                    url_page_title=result.get("url_page_title"),
                    url_page_date=datetime.fromisoformat(result.get("url_page_date"))
                    if result.get("url_page_date")
                    else None,
                    summary=result.get("summary"),
                )
                db.add(db_result)

            # Commit each batch
            db.commit()
            logger.info(
                f"Saved batch of {len(batch)} results (batch {i//BATCH_SIZE + 1})"
            )

        logger.info(f"Saved all search results to database with query_id: {query_id}")
        return query_id

    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {str(e)}")
        # If a database error occurs, return a default query_id to prevent the API from failing
        # This is a fallback to ensure the API can still return results even if DB save fails
        return str(uuid.uuid4())


def translate_search_results_with_openai(
    results: List[Dict[str, Any]],
    target_language: str,
    target_language_name: str = None,
) -> List[Dict[str, Any]]:
    """
    Translate search results using OpenAI API in a single call.

    Args:
        results: List of search result dictionaries
        target_language: Target language code
        target_language_name: Target language name (optional)

    Returns:
        List of translated search result dictionaries
    """
    try:
        # Get OpenAI API key
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not configured")

        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)

        # Get target language name
        if not target_language_name:
            target_language_name = SUPPORTED_LANGUAGES_MAP.get(
                target_language, target_language
            )

        # Collect all texts to translate and create mapping
        texts_to_translate = []
        text_mapping = []

        for index, result in enumerate(results):
            # Translate title if it exists and is meaningful
            title = result.get("url_page_title") or result.get("title")
            if title and title != "No title available" and title.strip():
                texts_to_translate.append(title)
                text_mapping.append(
                    {
                        "result_index": index,
                        "field": "url_page_title",
                        "text_index": len(texts_to_translate) - 1,
                    }
                )

            # Translate description if it exists
            description = result.get("description") or result.get("summary")
            if description and description.strip():
                texts_to_translate.append(description)
                text_mapping.append(
                    {
                        "result_index": index,
                        "field": "description",
                        "text_index": len(texts_to_translate) - 1,
                    }
                )

            # Translate content title if it exists
            if (
                "content" in result
                and result["content"]
                and result["content"].get("title")
            ):
                content_title = result["content"]["title"]
                if content_title and content_title.strip():
                    texts_to_translate.append(content_title)
                    text_mapping.append(
                        {
                            "result_index": index,
                            "field": "content_title",
                            "text_index": len(texts_to_translate) - 1,
                        }
                    )

            # Translate content summary if it exists
            if (
                "content" in result
                and result["content"]
                and result["content"].get("summary")
            ):
                content_summary = result["content"]["summary"]
                if content_summary and content_summary.strip():
                    texts_to_translate.append(content_summary)
                    text_mapping.append(
                        {
                            "result_index": index,
                            "field": "content_summary",
                            "text_index": len(texts_to_translate) - 1,
                        }
                    )

        # If no texts to translate, return original results
        if not texts_to_translate:
            logger.warning("No translatable text found in results")
            return results

        logger.info(
            f"Translating {len(texts_to_translate)} texts to {target_language_name}"
        )

        # Create the prompt for OpenAI
        prompt = f"""Translate the following texts to {target_language_name}.
Return the translations as a JSON array in the same order as the input texts.
Preserve the meaning and tone, and make the translations natural and readable.
Keep any HTML tags or special formatting intact.

Texts to translate:
{chr(10).join([f"{i+1}. {text}" for i, text in enumerate(texts_to_translate)])}

Return only the JSON array of translated texts, nothing else."""

        # Make the OpenAI API call
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional translator. Always respond with a valid JSON array of translated texts in the same order as provided. Preserve formatting and HTML tags.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4000,
        )

        # Parse the response
        translated_content = response.choices[0].message.content.strip()

        try:
            # Parse the JSON response
            translated_texts = json.loads(translated_content)

            if not isinstance(translated_texts, list):
                raise ValueError("OpenAI response is not a JSON array")

            if len(translated_texts) != len(texts_to_translate):
                raise ValueError(
                    f"Translation count mismatch: expected {len(texts_to_translate)}, got {len(translated_texts)}"
                )

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse OpenAI response as JSON: {translated_content}"
            )
            raise ValueError(f"Invalid JSON response from OpenAI: {e}")

        # Create translated results by copying original and updating translated fields
        translated_results = []
        for result in results:
            translated_result = result.copy()
            translated_results.append(translated_result)

        # Map translations back to the results
        for mapping in text_mapping:
            result_index = mapping["result_index"]
            field = mapping["field"]
            text_index = mapping["text_index"]
            translated_text = translated_texts[text_index]

            if field == "url_page_title":
                translated_results[result_index]["url_page_title"] = translated_text
            elif field == "description":
                translated_results[result_index]["description"] = translated_text
            elif field == "content_title":
                if "content" not in translated_results[result_index]:
                    translated_results[result_index]["content"] = {}
                translated_results[result_index]["content"]["title"] = translated_text
            elif field == "content_summary":
                if "content" not in translated_results[result_index]:
                    translated_results[result_index]["content"] = {}
                translated_results[result_index]["content"]["summary"] = translated_text

        logger.info(f"Successfully translated {len(translated_texts)} texts")
        return translated_results

    except Exception as e:
        logger.error(f"Error translating search results: {str(e)}")
        raise e


# Enhanced API endpoint to retrieve results with content info for a specific query
@app.get("/api/search-results/{query_id}")
async def get_search_results(query_id: str, db: Session = Depends(get_db)):
    """Get search results for a specific query ID with enhanced content."""
    try:
        # Get the query
        query = (
            db.query(ImageSearchQuery)
            .filter(ImageSearchQuery.query_id == query_id)
            .first()
        )
        if not query:
            logger.error(f"Search query not found: {query_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Search query not found: {query_id}",
            )

        # Get the results
        results = (
            db.query(ImageSearchResult)
            .filter(ImageSearchResult.query_id == query_id)
            .all()
        )
        logger.info(f"Retrieved {len(results)} results for query: {query_id}")

        # Format results with content info
        formatted_results = []
        for result in results:
            result_data = {
                "image_path": result.image_path,
                "url": result.url,
                "related_percentage": result.related_percentage,
                "url_page_title": result.url_page_title,
                "url_page_date": result.url_page_date.isoformat()
                if result.url_page_date
                else None,
                "summary": result.summary,
            }

            # Add enhanced content info if processed
            if result.content_processed:
                result_data["content"] = {
                    "title": result.content_title,
                    "date": result.content_date,
                    "publisher": result.content_publisher,
                    "summary": result.content_summary,
                    "processed_at": result.content_processed_at.isoformat()
                    if result.content_processed_at
                    else None,
                }

            formatted_results.append(result_data)

        return {
            "query": {
                "query_id": query.query_id,
                "image_url": query.image_url,
                "tweet_id": query.tweet_id,
                "search_time": query.search_time,
                "created_at": query.created_at.isoformat(),
            },
            "results": formatted_results,
            "count": len(results),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving search results: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


# Add an endpoint to manually process URLs for a query
@app.post("/api/process-urls/{query_id}")
async def process_urls(
    query_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    """Manually trigger processing of all URLs for a specific query."""
    try:
        # Get the query
        query = (
            db.query(ImageSearchQuery)
            .filter(ImageSearchQuery.query_id == query_id)
            .first()
        )
        if not query:
            logger.error(f"Search query not found: {query_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Search query not found: {query_id}",
            )

        # Add background task to process all URLs
        background_tasks.add_task(process_all_urls_for_query, query_id, query.tweet_id)
        logger.info(
            f"Manually added background task to process URLs for query ID: {query_id}"
        )

        return {
            "status": "success",
            "message": f"URL processing started for query ID: {query_id}",
            "query_id": query_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting URL processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error: {str(e)}"
        )


# Add endpoint to manually process a single URL
@app.post("/api/process-url/{result_id}")
async def process_url_endpoint(result_id: int, db: Session = Depends(get_db)):
    """
    Manually process a URL for a specific search result.
    """
    try:
        # Get the search result
        result = (
            db.query(ImageSearchResult)
            .filter(ImageSearchResult.id == result_id)
            .first()
        )
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Search result not found: {result_id}",
            )

        # Get the query to get the tweet_id
        query = (
            db.query(ImageSearchQuery)
            .filter(ImageSearchQuery.query_id == result.query_id)
            .first()
        )
        if not query:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Search query not found for result: {result_id}",
            )

        # Process the URL
        success = process_and_update_url_content(
            db, result_id, result.url, query.tweet_id
        )

        if success:
            return {
                "status": "success",
                "message": f"URL processed successfully for result ID: {result_id}",
                "result_id": result_id,
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to process URL for result ID: {result_id}",
                "result_id": result_id,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error: {str(e)}"
        )


@app.post("/api/image-search", response_model=ImageSearchResponse)
async def search_image(
    request: ImageSearchRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Search for similar images on the web using Google Cloud Vision API.
    """
    start_time = time.time()
    logger.info(
        f"Processing image search request - URL: {request.image_url}, Tweet ID: {request.tweet_id}"
    )

    # Validate tweet_id
    if not request.tweet_id:
        logger.error("Missing required field: tweet_id")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="tweet_id is required"
        )

    # Download the image from the URL
    tmp_file_path, err = download_image_from_url(request.image_url)
    if err:
        logger.error(f"Image download failed: {str(err)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download image: {str(err)}",
        )

    try:
        # Load the original image using OpenCV for comparison
        original_image = cv2.imread(tmp_file_path)
        if original_image is None:
            logger.error("Failed to read original image")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to read original image",
            )

        # Search for similar images
        domains, err = request_google_cloud_vision_api(tmp_file_path, request.image_url)
        if err:
            logger.error(f"Vision API error: {str(err)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Google Cloud Vision API error: {str(err)}",
            )

        logger.info(f"Found {len(domains)} matching images")

        # Download each matching image and calculate real similarity percentage
        for domain in domains:
            try:
                # Download the matching image
                matching_tmp_path, download_err = download_image_from_url(
                    domain.image_path
                )
                if download_err:
                    logger.warning(
                        f"Could not download matching image: {domain.image_path}"
                    )
                    continue

                try:
                    # Calculate similarity
                    matching_image = cv2.imread(matching_tmp_path)
                    if matching_image is not None:
                        similarity = _calculate_related_percentage(
                            tmp_file_path=matching_tmp_path, image=original_image
                        )

                        if similarity is not None:
                            # Convert to percentage and round to 2 decimal places
                            domain.related_percentage = round(similarity * 100, 2)
                finally:
                    # Clean up the matching image temporary file
                    try:
                        if matching_tmp_path:
                            os.unlink(matching_tmp_path)
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Error processing matching image: {str(e)}")

        # Update domains with metadata
        fetch_metadata = True
        if fetch_metadata:
            domains = update_domain_with_metadata(domains, limit=RESULTS_INT)

        # Ensure all domains have at least a default title
        for domain in domains:
            if domain.url_page_title is None:
                parsed_url = urlparse(domain.url)
                domain.url_page_title = f"Content from {parsed_url.netloc}"

        # Convert domains to dictionaries for JSON serialization
        results = [domain.to_dict() for domain in domains]

        # Calculate search time
        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.2f} seconds")

        # Save to database - use try/except to handle potential DB errors
        query_id = None
        try:
            # Limit the number of results to save if there are too many
            max_results_to_save = (
                50  # Adjust this number based on what your DB can handle
            )
            if len(results) > max_results_to_save:
                logger.warning(
                    f"Limiting results to save from {len(results)} to {max_results_to_save}"
                )
                results_to_save = results[:max_results_to_save]
            else:
                results_to_save = results

            query_id = save_search_results_to_db(
                db, request.image_url, request.tweet_id, search_time, results_to_save
            )

            # Add background task to process all URLs with extended websearch
            background_tasks.add_task(
                process_all_urls_for_query, query_id, request.tweet_id
            )
            logger.info(
                f"Added background task to process URLs for query ID: {query_id}"
            )

        except Exception as e:
            logger.error(f"Failed to save to database: {str(e)}")
            # Generate a temporary query_id so the response can still work
            query_id = str(uuid.uuid4())
            logger.info(f"Using temporary query_id: {query_id} due to database error")

        # Ensure query_id is never None to prevent validation errors
        if query_id is None:
            query_id = str(uuid.uuid4())
            logger.info(f"Generated fallback query_id: {query_id}")

        # Return the response
        return ImageSearchResponse(
            results=results,
            count=len(results),
            search_time=search_time,
            query_id=query_id,
        )

    finally:
        # Clean up the temporary file
        try:
            if tmp_file_path:
                os.unlink(tmp_file_path)
        except Exception as e:
            logger.error(f"Error cleaning up temporary file: {str(e)}")


@app.get("/api/search-history")
async def get_search_history(db: Session = Depends(get_db)):
    """Get the search history."""
    try:
        queries = (
            db.query(ImageSearchQuery)
            .order_by(ImageSearchQuery.created_at.desc())
            .all()
        )
        logger.info(f"Retrieved {len(queries)} search history records")
        return [
            {
                "query_id": query.query_id,
                "image_url": query.image_url,
                "tweet_id": query.tweet_id,
                "search_time": query.search_time,
                "created_at": query.created_at.isoformat(),
                "result_count": len(query.results),
            }
            for query in queries
        ]
    except Exception as e:
        logger.error(f"Error retrieving search history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


@app.post("/api/translate", response_model=TranslateResponse)
async def translate_search_results(request: TranslateRequest):
    """
    Translate search results using OpenAI API.

    This endpoint takes search results in the same format as returned by the image search
    and translates all translatable text fields to the specified target language.
    """
    try:
        # Validate target language
        if request.target_language not in SUPPORTED_LANGUAGES_MAP:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported target language: {request.target_language}. Supported languages: {list(SUPPORTED_LANGUAGES_MAP.keys())}",
            )

        # Get target language name
        target_language_name = (
            request.target_language_name
            or SUPPORTED_LANGUAGES_MAP[request.target_language]
        )

        logger.info(
            f"Starting translation of {len(request.results)} results to {target_language_name}"
        )

        # Translate the results
        translated_results = translate_search_results_with_openai(
            results=request.results,
            target_language=request.target_language,
            target_language_name=target_language_name,
        )

        # Ensure proper UTF-8 encoding for the response
        response_data = TranslateResponse(
            translated_results=translated_results,
            original_count=len(request.results),
            translated_count=len(translated_results),
            target_language=request.target_language,
            target_language_name=target_language_name,
        )

        logger.info(
            f"Successfully translated {len(request.results)} results to {target_language_name}"
        )
        return response_data

    except ValueError as e:
        logger.error(f"Translation validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}",
        )


# Also add this to fix UTF-8 encoding in the OpenAI translation function
def translate_search_results_with_openai(
    results: List[Dict[str, Any]],
    target_language: str,
    target_language_name: str = None,
) -> List[Dict[str, Any]]:
    """
    Translate search results using OpenAI API with batching to handle large requests.

    Args:
        results: List of search result dictionaries
        target_language: Target language code
        target_language_name: Target language name (optional)

    Returns:
        List of translated search result dictionaries
    """
    try:
        # Get OpenAI API key
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not configured")

        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)

        # Get target language name
        if not target_language_name:
            target_language_name = SUPPORTED_LANGUAGES_MAP.get(
                target_language, target_language
            )

        # Collect all texts to translate and create mapping
        texts_to_translate = []
        text_mapping = []

        for index, result in enumerate(results):
            # Translate title if it exists and is meaningful
            title = result.get("url_page_title") or result.get("title")
            if title and title != "No title available" and title.strip():
                texts_to_translate.append(title)
                text_mapping.append(
                    {
                        "result_index": index,
                        "field": "url_page_title",
                        "text_index": len(texts_to_translate) - 1,
                    }
                )

            # Translate description if it exists
            description = result.get("description") or result.get("summary")
            if description and description.strip():
                texts_to_translate.append(description)
                text_mapping.append(
                    {
                        "result_index": index,
                        "field": "description",
                        "text_index": len(texts_to_translate) - 1,
                    }
                )

            # Translate content title if it exists
            if (
                "content" in result
                and result["content"]
                and result["content"].get("title")
            ):
                content_title = result["content"]["title"]
                if content_title and content_title.strip():
                    texts_to_translate.append(content_title)
                    text_mapping.append(
                        {
                            "result_index": index,
                            "field": "content_title",
                            "text_index": len(texts_to_translate) - 1,
                        }
                    )

            # Translate content summary if it exists
            if (
                "content" in result
                and result["content"]
                and result["content"].get("summary")
            ):
                content_summary = result["content"]["summary"]
                if content_summary and content_summary.strip():
                    texts_to_translate.append(content_summary)
                    text_mapping.append(
                        {
                            "result_index": index,
                            "field": "content_summary",
                            "text_index": len(texts_to_translate) - 1,
                        }
                    )

        # If no texts to translate, return original results
        if not texts_to_translate:
            logger.warning("No translatable text found in results")
            return results

        logger.info(
            f"Translating {len(texts_to_translate)} texts to {target_language_name}"
        )

        # Batch processing to avoid token limits
        BATCH_SIZE = 50
        all_translated_texts = []

        for i in range(0, len(texts_to_translate), BATCH_SIZE):
            batch_texts = texts_to_translate[i : i + BATCH_SIZE]
            batch_number = (i // BATCH_SIZE) + 1
            total_batches = (len(texts_to_translate) + BATCH_SIZE - 1) // BATCH_SIZE

            logger.info(
                f"Processing batch {batch_number}/{total_batches} ({len(batch_texts)} texts)"
            )

            try:
                # Create the prompt for this batch
                prompt = f"""Translate the following {len(batch_texts)} texts to {target_language_name}.
Return the translations as a JSON array in the same order as the input texts.
Preserve the meaning and tone, and make the translations natural and readable.
Keep any HTML tags or special formatting intact.
Use proper Unicode characters and ensure correct encoding for accented characters.

Texts to translate:
{chr(10).join([f"{idx+1}. {text}" for idx, text in enumerate(batch_texts)])}

Return only the JSON array of translated texts with proper Unicode encoding, nothing else."""

                # Make the OpenAI API call for this batch
                response = client.chat.completions.create(
                    model="gpt-4o",  # Latest and fastest model with 128K context
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional translator. Always respond with a valid JSON array of translated texts in the same order as provided. Use proper Unicode characters and ensure correct UTF-8 encoding for all accented characters and special symbols.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=8000,  # Much larger context allows more tokens
                )

                # Parse the response with proper UTF-8 handling
                translated_content = response.choices[0].message.content.strip()

                # Clean up the response to ensure it's valid JSON
                # Remove any potential markdown formatting
                if translated_content.startswith("```json"):
                    translated_content = translated_content[7:]
                if translated_content.endswith("```"):
                    translated_content = translated_content[:-3]
                translated_content = translated_content.strip()

                try:
                    # Parse the JSON response with UTF-8 encoding
                    batch_translated_texts = json.loads(translated_content)

                    if not isinstance(batch_translated_texts, list):
                        raise ValueError(
                            f"Batch {batch_number}: OpenAI response is not a JSON array"
                        )

                    if len(batch_translated_texts) != len(batch_texts):
                        # If count mismatch, try retry with smaller batch
                        logger.warning(
                            f"Batch {batch_number}: Count mismatch ({len(batch_translated_texts)}/{len(batch_texts)}), retrying with smaller batches"
                        )

                        # Split into 2 smaller batches and retry
                        mid_point = len(batch_texts) // 2
                        batch_1 = batch_texts[:mid_point]
                        batch_2 = batch_texts[mid_point:]

                        retry_translated_texts = []

                        # Retry batch 1
                        for retry_batch in [batch_1, batch_2]:
                            if len(retry_batch) == 0:
                                continue

                            retry_prompt = f"""Translate the following {len(retry_batch)} texts to {target_language_name}.
Return the translations as a JSON array in the same order as the input texts.
Preserve the meaning and tone, and make the translations natural and readable.

Texts to translate:
{chr(10).join([f"{idx+1}. {text}" for idx, text in enumerate(retry_batch)])}

Return only the JSON array of translated texts, nothing else."""

                            try:
                                retry_response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": "You are a professional translator. Always respond with a valid JSON array of translated texts in the same order as provided.",
                                        },
                                        {"role": "user", "content": retry_prompt},
                                    ],
                                    temperature=0.3,
                                    max_tokens=3000,  # Smaller limit for retry
                                )

                                retry_content = retry_response.choices[
                                    0
                                ].message.content.strip()
                                if retry_content.startswith("```json"):
                                    retry_content = retry_content[7:]
                                if retry_content.endswith("```"):
                                    retry_content = retry_content[:-3]
                                retry_content = retry_content.strip()

                                retry_translations = json.loads(retry_content)
                                if len(retry_translations) == len(retry_batch):
                                    retry_translated_texts.extend(retry_translations)
                                else:
                                    logger.warning(
                                        f"Retry batch still has count mismatch, using original texts"
                                    )
                                    retry_translated_texts.extend(retry_batch)

                            except Exception as retry_e:
                                logger.error(
                                    f"Retry failed: {str(retry_e)}, using original texts"
                                )
                                retry_translated_texts.extend(retry_batch)

                        if len(retry_translated_texts) == len(batch_texts):
                            batch_translated_texts = retry_translated_texts
                            logger.info(
                                f"Successfully recovered batch {batch_number} with retry"
                            )
                        else:
                            raise ValueError(
                                f"Retry also failed for batch {batch_number}"
                            )

                    if len(batch_translated_texts) != len(batch_texts):
                        raise ValueError(
                            f"Batch {batch_number}: Translation count mismatch: expected {len(batch_texts)}, got {len(batch_translated_texts)}"
                        )

                    all_translated_texts.extend(batch_translated_texts)
                    logger.info(
                        f"Successfully translated batch {batch_number}/{total_batches}"
                    )

                except json.JSONDecodeError as e:
                    logger.error(
                        f"Batch {batch_number}: Failed to parse OpenAI response as JSON: {translated_content[:500]}..."
                    )
                    # For failed batches, use original texts as fallback
                    logger.warning(
                        f"Batch {batch_number}: Using original texts as fallback"
                    )
                    all_translated_texts.extend(batch_texts)

            except Exception as e:
                logger.error(f"Batch {batch_number}: API call failed: {str(e)}")
                # For failed batches, use original texts as fallback
                logger.warning(
                    f"Batch {batch_number}: Using original texts as fallback"
                )
                all_translated_texts.extend(batch_texts)

            # Small delay between batches to avoid rate limiting
            if i + BATCH_SIZE < len(texts_to_translate):
                time.sleep(0.5)

        # Verify we have all translations
        if len(all_translated_texts) != len(texts_to_translate):
            logger.error(
                f"Total translation count mismatch: expected {len(texts_to_translate)}, got {len(all_translated_texts)}"
            )
            raise ValueError("Failed to translate all texts")

        # Create translated results by copying original and updating translated fields
        translated_results = []
        for result in results:
            # Deep copy to avoid modifying original
            translated_result = json.loads(json.dumps(result, ensure_ascii=False))
            translated_results.append(translated_result)

        # Map translations back to the results with proper UTF-8 handling
        for mapping in text_mapping:
            result_index = mapping["result_index"]
            field = mapping["field"]
            text_index = mapping["text_index"]

            if text_index < len(all_translated_texts):
                translated_text = all_translated_texts[text_index]

                # Ensure proper UTF-8 encoding
                if isinstance(translated_text, str):
                    translated_text = translated_text.encode("utf-8").decode("utf-8")

                if field == "url_page_title":
                    translated_results[result_index]["url_page_title"] = translated_text
                elif field == "description":
                    translated_results[result_index]["description"] = translated_text
                elif field == "content_title":
                    if "content" not in translated_results[result_index]:
                        translated_results[result_index]["content"] = {}
                    translated_results[result_index]["content"][
                        "title"
                    ] = translated_text
                elif field == "content_summary":
                    if "content" not in translated_results[result_index]:
                        translated_results[result_index]["content"] = {}
                    translated_results[result_index]["content"][
                        "summary"
                    ] = translated_text

        logger.info(
            f"Successfully translated {len(all_translated_texts)} texts in {(len(texts_to_translate) + BATCH_SIZE - 1) // BATCH_SIZE} batches"
        )
        return translated_results

    except Exception as e:
        logger.error(f"Error translating search results: {str(e)}")
        raise e


# endpoint to get supported languages
@app.get("/api/supported-languages")
async def get_supported_languages():
    """Get list of supported languages for translation."""
    return {
        "languages": [
            {"code": code, "name": name}
            for code, name in SUPPORTED_LANGUAGES_MAP.items()
        ]
    }


# basic health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "database": "disconnected", "error": str(e)}


# Run the FastAPI app with Uvicorn if executed directly
if __name__ == "__main__":
    logger.info("Starting Image Search API server")
    uvicorn.run(app, host="0.0.0.0", port=5500)
