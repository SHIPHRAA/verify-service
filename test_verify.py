import os
import requests
import uuid
import sys

API_URL = "http://localhost:8000/image-analysis/"
AUTH_TOKEN = "AAAAAAAAAAAAAAAAAAAAAMYyxwEAAAAAz0pJfDJzr2LNIenV53T%2F1h0dcqs%3DDWus8RaCvnUirgmu1lkttCHq1j0ft3uBGM1Lg6sTTJdlgqRRmA"

def verify_image(file_path):
    payload = {
        "check_target_file_id": str(uuid.uuid4()),
        "file_path": file_path
    }

    headers = {
        "Content-Type": "application/json",
        "Authentication": AUTH_TOKEN
    }

    response = requests.post(API_URL, json=payload, headers=headers)

    if response.ok:
        print(f"✅ {file_path} ->", response.json())
    else:
        print(f"❌ {file_path} -> Error {response.status_code}: {response.text}")

def process_directory(directory_path):
    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(supported_extensions):
                full_path = os.path.join(root, file)
                verify_image(full_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_verify.py <image_directory>")
    else:
        process_directory(sys.argv[1])
