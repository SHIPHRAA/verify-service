import os
import openai
import base64
import sys
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image_with_gpt4o(image_path):
    image_base64 = encode_image_to_base64(image_path)

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You're an AI image authenticity detector. Analyze this image "
                            "and estimate the percentage likelihood that it is AI-generated. "
                            "Just give a short answer like: 'AI-generated: 85%' or 'Real image: 10% AI-likelihood'."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ],
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

def process_directory(directory_path):
    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(supported_extensions):
                full_path = os.path.join(root, file)
                print(f"\n🔍 {full_path} →")
                try:
                    result = analyze_image_with_gpt4o(full_path)
                    print(f"📊 GPT-4o says: {result}")
                except Exception as e:
                    print(f"❌ Error analyzing {full_path}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gpt4o_image_analysis.py <image_directory>")
    else:
        process_directory(sys.argv[1])
