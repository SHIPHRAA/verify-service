import os
import sys
import json
import uuid
import base64
import requests
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

API_URL = "http://localhost:8000/image-analysis/"
AUTH_TOKEN = "AAAAAAAAAAAAAAAAAAAAAMYyxwEAAAAAz0pJfDJzr2LNIenV53T%2F1h0dcqs%3DDWus8RaCvnUirgmu1lkttCHq1j0ft3uBGM1Lg6sTTJdlgqRRmA"

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_with_gpt4o(image_path):
    image_base64 = encode_image_to_base64(image_path)

    def ask(prompt):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": prompt },
                        {
                            "type": "image_url",
                            "image_url": { "url": f"data:image/jpeg;base64,{image_base64}" }
                        }
                    ],
                }
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    initial_prompt = (
        "You are an AI-generated image detector. Estimate the likelihood this image was AI-generated. "
        "Reply only with: 'AI-generated: XX%' or 'Real image: XX% AI-likelihood'. No disclaimers or uncertainty.""
    )
    result = ask(initial_prompt)

    if any(word in result.lower() for word in ["can't", "unsure", "unable"]):
        retry_prompt = "Be assertive. Force a decision and return a percentage likelihood that the image is AI-generated."
        result = ask(retry_prompt)

    return result

def analyze_with_verify_service(image_path, check_id):
    payload = {
        "check_target_file_id": check_id,
        "file_path": image_path
    }
    headers = {
        "Content-Type": "application/json",
        "Authentication": AUTH_TOKEN
    }
    response = requests.post(API_URL, json=payload, headers=headers)
    return response.json() if response.ok else {"error": response.text}

def process_directory(image_dir):
    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(supported_extensions):
                full_path = os.path.join(root, file)
                check_id = str(uuid.uuid4())

                print(f"\n Image: {full_path}")
                print(f"Image ID: {check_id}")

                try:
                    verify_result = analyze_with_verify_service(full_path, check_id)
                    gpt_result = analyze_with_gpt4o(full_path)

                    print("\n Verify Service:")
                    print(json.dumps(verify_result.get("result", verify_result), indent=2))

                    print("\n GPT-4o:")
                    print(gpt_result)

                except Exception as e:
                    print(f"❌ Error processing {full_path}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_verify_and_gpt4o.py <image_directory>")
    else:
        process_directory(sys.argv[1])
