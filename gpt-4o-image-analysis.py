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

def call_gpt4o_with_prompt(image_base64, prompt_text):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
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

def analyze_image_with_gpt4o(image_path):
    image_base64 = encode_image_to_base64(image_path)

    primary_prompt = (
        "You are an expert AI-generated image detector. Analyze the image provided and respond with a single line like "
        "'AI-generated: 90%' or 'Real image: 5% AI-likelihood'. Avoid vague responses. If unsure, still provide your best estimate."
    )

    response_text = call_gpt4o_with_prompt(image_base64, primary_prompt)

    if any(keyword in response_text.lower() for keyword in ["can't", "unable", "not sure", "unsure", "don't know"]):
        retry_prompt = (
            "Be assertive. Estimate how likely this image is AI-generated. Give your best guess. Respond only with: "
            "'AI-generated: XX%' or 'Real image: XX% AI-likelihood'. No disclaimers or uncertainty."
        )
        response_text = call_gpt4o_with_prompt(image_base64, retry_prompt)

    return response_text

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
