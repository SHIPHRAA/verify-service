import os
import json
import argparse
import requests
import re
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# Twitter API constants
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN", "")


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
    pattern = r"(?:twitter\.com|x\.com)/\w+/status/(\d+)"
    match = re.search(pattern, url)

    if match:
        return match.group(1)
    return None


def get_tweet_by_id(tweet_id):
    """
    Get tweet data using Twitter API

    Args:
        tweet_id (str): Tweet ID

    Returns:
        dict: Tweet data or error
    """
    url = f"https://api.twitter.com/2/tweets/{tweet_id}"

    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}

    params = {
        "expansions": "author_id",
        "tweet.fields": "created_at,text",
        "user.fields": "name,username",
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching tweet: {e}")
        if hasattr(e, "response") and e.response:
            print(f"API response: {e.response.text}")
        return {"error": f"Failed to fetch tweet: {e}"}


def format_tweet_info(tweet_data):
    """
    Format tweet data into a structured response

    Args:
        tweet_data (dict): Tweet data from Twitter API

    Returns:
        dict: Structured tweet info
    """
    try:
        # Check if there was an error
        if "error" in tweet_data:
            return {
                "title": "Failed to retrieve tweet",
                "date": None,
                "publisher": "Twitter/X",
                "summary": f"Error: {tweet_data['error']}",
            }

        # Extract the relevant information
        tweet = tweet_data["data"]
        user = tweet_data["includes"]["users"][0]

        # Format the date
        created_at = datetime.strptime(tweet["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
        formatted_date = created_at.strftime("%Y-%m-%d")

        # Create structured response in the same format as extract_article_info
        return {
            "title": f"Tweet by {user['name']} (@{user['username']})",
            "date": formatted_date,
            "publisher": "Twitter/X",
            "summary": tweet["text"],
        }
    except Exception as e:
        print(f"Error formatting tweet info: {e}")
        return {
            "title": "Failed to process tweet data",
            "date": None,
            "publisher": "Twitter/X",
            "summary": f"Error: Failed to process tweet data - {str(e)}",
        }


def extract_article_info(url, api_key):
    """
    Extract article information using Beautiful Soup for HTML parsing
    and OpenAI for summarization

    Args:
        url (str): URL of the article to analyze
        api_key (str): OpenAI API key

    Returns:
        dict: JSON response with article information
    """
    article_data = {"title": None, "date": None, "publisher": None, "summary": None}

    try:
        # Add user agent to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # Get the webpage content
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Check if the page requires JavaScript (common error messages)
        js_required = False
        js_indicators = [
            "JavaScript is not available",
            "JavaScript is disabled",
            "enable JavaScript",
            "browser does not support JavaScript",
            "Please enable JavaScript",
            "This page requires JavaScript",
        ]

        page_text = soup.get_text().lower()
        for indicator in js_indicators:
            if indicator.lower() in page_text:
                js_required = True
                break

        # If page requires JavaScript, return appropriate message
        if js_required:
            domain = urlparse(url).netloc
            if domain.startswith("www."):
                domain = domain[4:]
            domain_parts = domain.split(".")
            if len(domain_parts) >= 2:
                site_name = domain_parts[0].title()
            else:
                site_name = domain.title()

            article_data["title"] = f"Content not accessible (JavaScript required)"
            article_data["publisher"] = site_name
            article_data[
                "summary"
            ] = f"This article on {site_name} requires JavaScript to be accessed. The content is loaded dynamically and cannot be extracted with the current method. Consider using a browser automation tool like Selenium or Playwright to access this content."
            return article_data

        # Extract title
        title = None
        # Check for OpenGraph title tag
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title.get("content")
        # Check for Twitter title tag
        if not title:
            twitter_title = soup.find("meta", attrs={"name": "twitter:title"})
            if twitter_title and twitter_title.get("content"):
                title = twitter_title.get("content")
        # Check for headline tags
        if not title:
            headline = soup.find(
                ["h1", "h2"],
                class_=lambda c: c
                and ("headline" in c.lower() or "title" in c.lower()),
            )
            if headline:
                title = headline.get_text().strip()
        # Fallback to main heading
        if not title:
            main_heading = soup.find("h1")
            if main_heading:
                title = main_heading.get_text().strip()
        # Fallback to page title
        if not title and soup.title:
            title = soup.title.get_text().strip()

        article_data["title"] = title

        # Extract date - search for visible date text first
        date = None
        visible_date = None

        # Look for text that might contain a date with day of week
        for element in soup.find_all(["span", "div", "p", "time"]):
            text = element.get_text().strip()
            # Match patterns like "Tuesday 20 May 2025"
            weekday_match = re.search(
                r"([A-Za-z]+day\s+\d{1,2}\s+[A-Za-z]+\s+\d{4})", text
            )
            if weekday_match:
                visible_date = weekday_match.group(1)
                break

        # If found visible date with weekday, parse it
        if visible_date:
            parts = visible_date.split()
            if len(parts) >= 4:
                day = parts[1].zfill(2)  # Day is the second part
                month_name = parts[2]  # Month name is the third part
                year = parts[3]  # Year is the fourth part

                month_map = {
                    "january": "01",
                    "february": "02",
                    "march": "03",
                    "april": "04",
                    "may": "05",
                    "june": "06",
                    "july": "07",
                    "august": "08",
                    "september": "09",
                    "october": "10",
                    "november": "11",
                    "december": "12",
                }
                month = month_map.get(month_name.lower(), "01")
                date = f"{year}-{month}-{day}"

        # If no visible date with weekday, try other methods
        if not date:
            # Try schema.org datePublished
            schema_date = soup.find("meta", property="article:published_time")
            if schema_date and schema_date.get("content"):
                date = schema_date.get("content")
                # Extract just YYYY-MM-DD if longer
                iso_match = re.match(r"(\d{4}-\d{2}-\d{2})", date)
                if iso_match:
                    date = iso_match.group(1)

        # Try time tag
        if not date:
            time_tag = soup.find("time")
            if time_tag and time_tag.get("datetime"):
                datetime_value = time_tag.get("datetime")
                # Extract YYYY-MM-DD if in ISO format
                iso_match = re.match(r"(\d{4}-\d{2}-\d{2})", datetime_value)
                if iso_match:
                    date = iso_match.group(1)
                else:
                    date = time_tag.get_text().strip()

        # Try common date classes
        if not date:
            date_div = soup.find(
                ["div", "span", "p"],
                class_=lambda c: c
                and (
                    "date" in c.lower()
                    or "time" in c.lower()
                    or "published" in c.lower()
                ),
            )
            if date_div:
                date = date_div.get_text().strip()

        # If date is just a time format (HH:MM:SS), look for the date elsewhere
        if date and re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", date):
            # Save the time part
            time_part = date
            date = None

            # Search the entire page text again for a date format
            date_found = False
            for element in soup.find_all(["span", "div", "p", "time"]):
                text = element.get_text().strip()

                # Try various date formats
                date_patterns = [
                    # "Tuesday 20 May 2025"
                    r"([A-Za-z]+day\s+\d{1,2}\s+[A-Za-z]+\s+\d{4})",
                    # "20 May 2025"
                    r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
                    # "May 20, 2025"
                    r"([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                ]

                for pattern in date_patterns:
                    match = re.search(pattern, text)
                    if match:
                        date_text = match.group(1)

                        # Parse the date based on its format
                        if re.match(
                            r"[A-Za-z]+day\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}", date_text
                        ):
                            # Format: "Tuesday 20 May 2025"
                            parts = date_text.split()
                            day = parts[1].zfill(2)
                            month_name = parts[2]
                            year = parts[3]

                            month_map = {
                                "january": "01",
                                "february": "02",
                                "march": "03",
                                "april": "04",
                                "may": "05",
                                "june": "06",
                                "july": "07",
                                "august": "08",
                                "september": "09",
                                "october": "10",
                                "november": "11",
                                "december": "12",
                            }
                            month = month_map.get(month_name.lower(), "01")
                            date = f"{year}-{month}-{day}"
                            date_found = True
                            break

                        elif re.match(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}", date_text):
                            # Format: "20 May 2025"
                            parts = date_text.split()
                            day = parts[0].zfill(2)
                            month_name = parts[1]
                            year = parts[2]

                            month_map = {
                                "january": "01",
                                "february": "02",
                                "march": "03",
                                "april": "04",
                                "may": "05",
                                "june": "06",
                                "july": "07",
                                "august": "08",
                                "september": "09",
                                "october": "10",
                                "november": "11",
                                "december": "12",
                            }
                            month = month_map.get(month_name.lower(), "01")
                            date = f"{year}-{month}-{day}"
                            date_found = True
                            break

                        elif re.match(r"[A-Za-z]+\s+\d{1,2},?\s+\d{4}", date_text):
                            # Format: "May 20, 2025"
                            parts = re.match(
                                r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", date_text
                            )
                            if parts:
                                month_name = parts.group(1)
                                day = parts.group(2).zfill(2)
                                year = parts.group(3)

                                month_map = {
                                    "january": "01",
                                    "february": "02",
                                    "march": "03",
                                    "april": "04",
                                    "may": "05",
                                    "june": "06",
                                    "july": "07",
                                    "august": "08",
                                    "september": "09",
                                    "october": "10",
                                    "november": "11",
                                    "december": "12",
                                }
                                month = month_map.get(month_name.lower(), "01")
                                date = f"{year}-{month}-{day}"
                                date_found = True
                                break

                if date_found:
                    break

        article_data["date"] = date

        # Extract publisher
        publisher = None

        # Try OpenGraph site name
        og_site = soup.find("meta", property="og:site_name")
        if og_site and og_site.get("content"):
            publisher = og_site.get("content")

        # Try schema.org publisher
        if not publisher:
            schema_publisher = soup.find("meta", property="article:publisher")
            if schema_publisher and schema_publisher.get("content"):
                publisher = schema_publisher.get("content")

        # Fallback to domain name
        if not publisher:
            domain = urlparse(url).netloc
            # Remove www. if present
            if domain.startswith("www."):
                domain = domain[4:]
            # Convert to title case and remove TLD
            publisher = domain.split(".")[0].title()

        article_data["publisher"] = publisher

        # Extract main content for summarization
        main_content = ""

        # Try to find article content using common article containers
        article_container = soup.find(
            ["article", "div", "section"],
            class_=lambda c: c
            and (
                "article" in str(c).lower()
                or "content" in str(c).lower()
                or "story" in str(c).lower()
            ),
        )

        if article_container:
            # Get all paragraphs from the article container
            paragraphs = article_container.find_all("p")
            main_content = " ".join([p.get_text().strip() for p in paragraphs])
        else:
            # Fallback to all paragraphs
            paragraphs = soup.find_all("p")
            main_content = " ".join([p.get_text().strip() for p in paragraphs])

        # If main_content is empty or too short, try to get text from all div elements
        if len(main_content.strip()) < 50:
            # Try to find content in div elements
            content_divs = soup.find_all(
                "div",
                class_=lambda c: c
                and (
                    "text" in str(c).lower()
                    or "body" in str(c).lower()
                    or "content" in str(c).lower()
                ),
            )
            div_text = []
            for div in content_divs:
                # Skip navigation, header, footer, and sidebar content
                if div.find(
                    class_=lambda c: c
                    and (
                        "nav" in str(c).lower()
                        or "header" in str(c).lower()
                        or "footer" in str(c).lower()
                        or "sidebar" in str(c).lower()
                    )
                ):
                    continue
                div_text.append(div.get_text().strip())

            if div_text:
                main_content = " ".join(div_text)

        # If still empty, try with the entire body content
        if len(main_content.strip()) < 50:
            body = soup.find("body")
            if body:
                # Try to filter out navigation, header, footer elements
                for nav in body.find_all(["nav", "header", "footer"]):
                    nav.decompose()
                main_content = body.get_text().strip()

        # Use OpenAI for summarization only if we have content and API key
        if main_content and api_key:
            try:
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise summarization assistant. Create a concise, factual 40-word summary of the following article content. If the content appears to be fabricated or false news, mention that in your summary.",
                        },
                        {
                            "role": "user",
                            "content": f"Create a concise 40-word summary of this article content:\n\n{main_content[:4000]}",
                        },  # Limit to first 4000 chars
                    ],
                    max_tokens=100,
                )

                summary = response.choices[0].message.content.strip()
                # If summary is too short or just punctuation, provide a fallback
                if len(summary) < 5 or not any(c.isalpha() for c in summary):
                    summary = "Failed to generate a proper summary due to insufficient or inaccessible article content."
                article_data["summary"] = summary
            except Exception as e:
                print(f"OpenAI API error: {e}")
                # Fallback to first few sentences if API fails
                sentences = main_content.split(".")
                if len(sentences) >= 2:
                    article_data["summary"] = ". ".join(sentences[:2]).strip() + "."
                else:
                    article_data[
                        "summary"
                    ] = "Failed to generate a summary due to API error or insufficient content."
        else:
            # Fallback if no API key or content is too short
            if len(main_content.strip()) > 50:
                sentences = main_content.split(".")
                if len(sentences) >= 2:
                    article_data["summary"] = ". ".join(sentences[:2]).strip() + "."
                else:
                    article_data["summary"] = main_content[:150] + "..."
            else:
                article_data[
                    "summary"
                ] = "Failed to extract sufficient article content for summarization."

        return article_data

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return {
            "title": "Failed to fetch URL",
            "date": None,
            "publisher": urlparse(url).netloc.split(".")[0].title()
            if urlparse(url).netloc
            else "Unknown",
            "summary": f"Failed to fetch the URL: {e}",
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "title": "Error processing content",
            "date": None,
            "publisher": urlparse(url).netloc.split(".")[0].title()
            if urlparse(url).netloc
            else "Unknown",
            "summary": f"Failed to extract article info: {e}",
        }


def process_url(url, api_key=None):
    """
    Main function to process a URL (either Twitter/X or normal website)

    Args:
        url (str): URL to process
        api_key (str, optional): OpenAI API key for website summarization

    Returns:
        dict: Structured information about the content
    """
    # Check if it's a Twitter/X URL
    if is_twitter_url(url):
        # Extract the tweet ID and process with Twitter API
        tweet_id = extract_tweet_id(url)
        if tweet_id:
            tweet_data = get_tweet_by_id(tweet_id)
            return format_tweet_info(tweet_data)
        else:
            return {
                "title": "Invalid Twitter/X URL",
                "date": None,
                "publisher": "Twitter/X",
                "summary": "Could not extract tweet ID from the provided URL.",
            }
    else:
        # Process as a normal website
        return extract_article_info(url, api_key)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract information from websites and Twitter/X posts"
    )
    parser.add_argument("url", help="The URL to analyze")
    parser.add_argument(
        "--api-key",
        help="OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable",
    )
    parser.add_argument(
        "--twitter-token",
        help="Twitter Bearer Token. If not provided, will look for TWITTER_BEARER_TOKEN environment variable",
    )

    # Parse arguments
    args = parser.parse_args()

    # Get API key from arguments or environment variable
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    # Get Twitter token from arguments or environment variable
    twitter_token = args.twitter_token or os.environ.get("TWITTER_BEARER_TOKEN")
    if twitter_token:
        global TWITTER_BEARER_TOKEN
        TWITTER_BEARER_TOKEN = twitter_token

    # Process the URL
    result = process_url(args.url, api_key)

    # Print formatted JSON output
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
