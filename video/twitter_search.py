import os
import json
import argparse
import requests
import re
from datetime import datetime
from urllib.parse import urlparse
from dotenv import load_dotenv

# FastAPI imports
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

# Load environment variables FIRST
load_dotenv()

# Twitter API constants - with better fallback handling
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")

# Debug: Print token status (remove in production)
if TWITTER_BEARER_TOKEN:
    print(f"Token loaded successfully (length: {len(TWITTER_BEARER_TOKEN)})")
else:
    print("Warning: No Twitter Bearer Token found in environment")

# FastAPI Router
router = APIRouter(prefix="/twitter", tags=["twitter"])


# Pydantic models for API
class TwitterURLRequest(BaseModel):
    url: str
    bearer_token: Optional[str] = None


class MediaItem(BaseModel):
    type: Optional[str]
    preview_image_url: Optional[str]
    url: Optional[str]


class TwitterMediaResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    media: Optional[List[MediaItem]] = None
    media_count: Optional[int] = None
    message: Optional[str] = None


def debug_token_status():
    """Debug function to check token status"""
    token = TWITTER_BEARER_TOKEN or os.environ.get("TWITTER_BEARER_TOKEN")
    print(f"Debug - Token exists: {bool(token)}")
    if token:
        print(f"Debug - Token starts with: {token[:20]}...")
    else:
        print("Debug - No token found")
        print(f"Debug - Global TWITTER_BEARER_TOKEN: {bool(TWITTER_BEARER_TOKEN)}")
        print(
            f"Debug - Environment TWITTER_BEARER_TOKEN: {bool(os.environ.get('TWITTER_BEARER_TOKEN'))}"
        )
    return token


def get_bearer_token(provided_token=None):
    """
    Get bearer token from various sources with priority order

    Args:
        provided_token (str, optional): Token provided directly

    Returns:
        str: Bearer token or None if not found
    """
    # Priority order: provided > global > environment
    token = (
        provided_token or TWITTER_BEARER_TOKEN or os.environ.get("TWITTER_BEARER_TOKEN")
    )

    if not token:
        print("Debug: All token sources failed")
        print(f"- Provided token: {bool(provided_token)}")
        print(f"- Global token: {bool(TWITTER_BEARER_TOKEN)}")
        print(f"- Env token: {bool(os.environ.get('TWITTER_BEARER_TOKEN'))}")

    return token


def is_twitter_url(url):
    """
    Check if the URL is from Twitter/X

    Args:
        url (str): URL to check

    Returns:
        bool: True if URL is from Twitter/X, False otherwise
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    return domain in ["twitter.com", "x.com", "www.twitter.com", "www.x.com"]


def extract_tweet_id(url):
    """
    Extract tweet ID from Twitter/X URL

    Args:
        url (str): Twitter/X URL

    Returns:
        str: Tweet ID or None if not found
    """
    # Pattern to match tweet URLs from both twitter.com and x.com
    patterns = [
        r"(?:twitter\.com|x\.com)/\w+/status/(\d+)",
        r"(?:twitter\.com|x\.com)/i/status/(\d+)",
        r"status/(\d+)",
        r"statuses/(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def extract_media_info(api_response):
    """
    Extract specific media information from Twitter API response

    Args:
        api_response (dict): Raw Twitter API response

    Returns:
        dict: Simplified response with media info or error
    """
    try:
        if not api_response.get("_api_success"):
            return {
                "success": False,
                "error": api_response.get("error", "API request failed"),
                "media": None,
            }

        # Check if there's media in the response
        if "includes" not in api_response or "media" not in api_response["includes"]:
            return {
                "success": True,
                "error": None,
                "media": None,
                "message": "No media found in this tweet",
            }

        media_list = api_response["includes"]["media"]
        extracted_media = []

        for media_item in media_list:
            media_info = {
                "type": media_item.get("type"),
                "preview_image_url": media_item.get("preview_image_url"),
                "url": None,
            }

            # Handle different media types
            if media_item.get("type") == "video":
                # For videos, get the highest quality variant
                variants = media_item.get("variants", [])
                if variants:
                    # Filter MP4 variants and sort by bitrate
                    mp4_variants = [
                        v for v in variants if v.get("content_type") == "video/mp4"
                    ]
                    if mp4_variants:
                        # Get highest bitrate variant
                        best_variant = max(
                            mp4_variants, key=lambda x: x.get("bit_rate", 0)
                        )
                        media_info["url"] = best_variant.get("url")
                    else:
                        # Fallback to first variant if no MP4 found
                        media_info["url"] = variants[0].get("url")

            elif media_item.get("type") == "photo":
                media_info["url"] = media_item.get("url")

            elif media_item.get("type") == "animated_gif":
                variants = media_item.get("variants", [])
                if variants:
                    media_info["url"] = variants[0].get("url")

            extracted_media.append(media_info)

        return {
            "success": True,
            "error": None,
            "media": extracted_media,
            "media_count": len(extracted_media),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to extract media info: {str(e)}",
            "media": None,
        }


def get_tweet_by_id(tweet_id, bearer_token=None):
    """
    Get tweet data using Twitter API v2 - Returns processed media info

    Args:
        tweet_id (str): Tweet ID
        bearer_token (str, optional): Twitter Bearer Token

    Returns:
        dict: Processed response with media information
    """
    # Get bearer token with improved fallback handling
    token = get_bearer_token(bearer_token)

    if not token:
        return {
            "success": False,
            "error": "Twitter Bearer Token not provided. Set TWITTER_BEARER_TOKEN environment variable or pass as parameter.",
            "error_type": "missing_token",
            "media": None,
        }

    url = f"https://api.twitter.com/2/tweets/{tweet_id}"

    headers = {"Authorization": f"Bearer {token}", "User-Agent": "TwitterSearchBot/1.0"}

    # Get comprehensive tweet data with focus on media
    params = {
        "expansions": "attachments.media_keys",
        "tweet.fields": "attachments,created_at,public_metrics,text",
        "media.fields": "duration_ms,height,media_key,preview_image_url,type,url,width,public_metrics,alt_text,variants",
    }

    try:
        print(f"Making request to Twitter API for tweet ID: {tweet_id}")
        response = requests.get(url, headers=headers, params=params, timeout=10)

        # Parse response
        try:
            response_data = response.json()
        except:
            response_data = {"error": "Invalid JSON response from Twitter API"}

        # Add status info
        response_data["_api_status_code"] = response.status_code
        response_data["_api_success"] = response.status_code == 200

        # Add error interpretations
        if response.status_code == 401:
            response_data["error"] = "Unauthorized: Invalid Twitter Bearer Token"
        elif response.status_code == 403:
            response_data["error"] = "Forbidden: Tweet may be private or restricted"
        elif response.status_code == 404:
            response_data["error"] = "Tweet not found"
        elif response.status_code == 429:
            response_data["error"] = "Rate limit exceeded. Please try again later."

        print(f"Twitter API response status: {response.status_code}")
        if not response_data["_api_success"]:
            print(f"Twitter API error: {response_data.get('error', 'Unknown error')}")

        # Extract and return only media info
        return extract_media_info(response_data)

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timeout - Twitter API did not respond in time",
            "error_type": "timeout",
            "media": None,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}",
            "error_type": "request_exception",
            "media": None,
        }


def search_tweet(url_or_id, bearer_token=None):
    """
    Main function to search for tweet media information

    Args:
        url_or_id (str): Twitter URL or tweet ID
        bearer_token (str, optional): Twitter Bearer Token

    Returns:
        dict: Processed response with media information
    """
    # Check if input is a URL or direct tweet ID
    if url_or_id.startswith("http"):
        # It's a URL
        if not is_twitter_url(url_or_id):
            return {
                "success": False,
                "error": "The provided URL is not from Twitter/X",
                "error_type": "invalid_url",
                "media": None,
                "provided_url": url_or_id,
            }

        tweet_id = extract_tweet_id(url_or_id)
        if not tweet_id:
            return {
                "success": False,
                "error": "Could not extract tweet ID from the provided URL",
                "error_type": "invalid_twitter_url",
                "media": None,
                "provided_url": url_or_id,
            }
    else:
        # Assume it's a direct tweet ID
        tweet_id = url_or_id.strip()

        # Validate tweet ID (should be numeric)
        if not tweet_id.isdigit():
            return {
                "success": False,
                "error": "Tweet ID should be numeric",
                "error_type": "invalid_tweet_id",
                "media": None,
                "provided_id": tweet_id,
            }

    # Get media info from tweet
    return get_tweet_by_id(tweet_id, bearer_token)


def test_twitter_connection():
    """Test function to verify Twitter API connection"""
    token = get_bearer_token()
    if not token:
        return {"error": "No token available"}

    # Test with a simple API call
    url = "https://api.twitter.com/2/tweets/20"  # Twitter's early tweet
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "TwitterSearchBot/1.0"}

    try:
        response = requests.get(url, headers=headers, timeout=5)
        return {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "token_valid": response.status_code != 401,
        }
    except Exception as e:
        return {"error": f"Connection test failed: {str(e)}"}


# FastAPI Endpoints
@router.post("/media", response_model=TwitterMediaResponse)
async def get_twitter_media(request: TwitterURLRequest):
    """
    Extract media information from Twitter/X URLs

    Args:
        request: TwitterURLRequest containing the URL and optional bearer token

    Returns:
        TwitterMediaResponse with media information
    """
    try:
        # Call the search_tweet function
        result = search_tweet(request.url, request.bearer_token)

        # Convert media items to proper format if they exist
        if result.get("media"):
            media_items = [MediaItem(**item) for item in result["media"]]
            result["media"] = media_items

        return TwitterMediaResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process Twitter URL: {str(e)}"
        )


@router.get("/health")
async def twitter_health():
    """Health check for Twitter API functionality"""
    token_configured = bool(get_bearer_token())
    connection_test = (
        test_twitter_connection() if token_configured else {"error": "No token"}
    )

    return {
        "status": "Twitter API endpoint healthy",
        "bearer_token_configured": token_configured,
        "connection_test": connection_test,
    }


def main():
    """Command line interface for the Twitter media extractor."""
    parser = argparse.ArgumentParser(
        description="Extract media information from Twitter/X posts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python twitter_search.py "https://x.com/user/status/1918707351772618760"
  python twitter_search.py "https://twitter.com/user/status/1234567890"
  python twitter_search.py --tweet-id "1918707351772618760"
  python twitter_search.py --tweet-id "1918707351772618760" --token "your_bearer_token"

Environment Variables:
  TWITTER_BEARER_TOKEN    Your Twitter API Bearer Token

Note: This returns only media type, URL, and preview image URL.
        """,
    )

    # Create mutually exclusive group for URL or tweet ID
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("url", nargs="?", help="Twitter/X URL to analyze")
    group.add_argument("--tweet-id", "-t", help="Tweet ID to analyze")

    parser.add_argument(
        "--token", help="Twitter Bearer Token (overrides TWITTER_BEARER_TOKEN env var)"
    )
    parser.add_argument(
        "--pretty", "-p", action="store_true", help="Pretty print JSON output"
    )
    parser.add_argument("--save", "-s", help="Save response to file")
    parser.add_argument(
        "--test-connection", action="store_true", help="Test Twitter API connection"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Enable debug if requested
    if args.debug:
        print("=== DEBUG MODE ENABLED ===")
        debug_token_status()

    # Handle test connection
    if args.test_connection:
        print("Testing Twitter API connection...")
        test_result = test_twitter_connection()
        print(json.dumps(test_result, indent=2))
        return 0 if test_result.get("success") else 1

    # Update global token if provided via command line
    global TWITTER_BEARER_TOKEN
    if args.token:
        TWITTER_BEARER_TOKEN = args.token
        print("Using token provided via --token parameter")

    # Determine input (URL or tweet ID)
    input_value = args.url if args.url else args.tweet_id

    # Get bearer token
    bearer_token = get_bearer_token(args.token)

    if not bearer_token:
        print("Error: Twitter Bearer Token required!")
        print("Either:")
        print("1. Set TWITTER_BEARER_TOKEN environment variable")
        print("2. Use --token parameter")
        print("\nTo get a Bearer Token:")
        print("1. Go to https://developer.twitter.com/")
        print("2. Create a developer account and app")
        print("3. Generate a Bearer Token in your app settings")
        return 1

    if args.debug:
        print("=== CONNECTION TEST ===")
        test_result = test_twitter_connection()
        print(f"Connection test: {test_result}")
        print("=== END CONNECTION TEST ===\n")

    # Search for tweet media info
    result = search_tweet(input_value, bearer_token)

    # Save to file if requested
    if args.save:
        try:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Response saved to: {args.save}")
        except Exception as e:
            print(f"Failed to save file: {e}")

    # Output results
    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))

    # Return appropriate exit code
    return 0 if result.get("success") else 1


# Function to be used when importing as module
def process_url(url, api_key=None):
    """
    Process a Twitter URL and return media info
    This function matches the signature expected by your main application

    Args:
        url (str): Twitter URL to process
        api_key (str): Not used for Twitter API (uses bearer token instead)

    Returns:
        dict: Processed response with media information
    """
    # Use the global bearer token or try to get it from environment
    bearer_token = get_bearer_token()

    if not bearer_token:
        print("Warning: No Twitter Bearer Token available for process_url")
        return {
            "success": False,
            "error": "Twitter Bearer Token not configured",
            "media": None,
        }

    return search_tweet(url, bearer_token)


if __name__ == "__main__":
    # Test connection on startup if token is available
    if get_bearer_token():
        print("Testing Twitter connection on startup...")
        test_result = test_twitter_connection()
        if test_result.get("success"):
            print("✓ Twitter API connection successful")
        else:
            print(f"✗ Twitter API connection failed: {test_result}")
        print()

    exit(main())
