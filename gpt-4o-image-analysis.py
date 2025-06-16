import os
import openai
import base64
import sys
import json
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_gpt4o_with_prompt(image_base64, prompt_text):
    try:
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
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_image_modifications(image_path):
    image_base64 = encode_image_to_base64(image_path)

    analysis_prompt = """
            You are an expert digital forensics analyst specializing in AI-generated content and image manipulation detection. Analyze the provided image and return your findings in JSON format only.
 
            Perform a comprehensive MODIFICATION ANALYSIS to determine authenticity and detect potential manipulation:
 
            - Facial analysis: Check for natural skin textures, symmetrical features, realistic aging patterns, proper facial proportions
            - Lighting consistency: Verify shadows match light sources, natural light distribution, realistic reflections
            - Background integrity: Look for seamless edges, realistic perspectives, properly anchored objects, natural depth of field
            - Image quality uniformity: Assess consistent resolution and clarity throughout the image, natural compression artifacts
            - Anatomical accuracy: Verify correct proportions, proper finger count, natural limb positioning, realistic body mechanics
            - Technical authenticity: Check for natural compression patterns, appropriate grain/noise levels, consistent metadata
            - Visual coherence: Ensure style consistency, natural color transitions, realistic material textures, proper physics
            - AI generation indicators: Look for telltale signs of AI generation such as unusual artifacts, impossible geometry, or synthetic patterns
            - Digital manipulation signs: Detect evidence of photo editing, compositing, or post-processing modifications
            - Overall assessment: Determine if image appears naturally photographed or shows signs of AI generation or digital editing
 
            Return ONLY a valid JSON object with this exact structure:
 
            {
                "modification_analysis": "<A detailed assessment of potential AI modification or digital manipulation. Describe any suspicious indicators such as facial inconsistencies, lighting anomalies, background artifacts, anatomical impossibilities, digital artifacts, AI generation patterns, or consistency issues. Also mention natural indicators that suggest authenticity. Provide your overall assessment of whether the image appears to be AI-generated, digitally modified, or authentic, along with your confidence level and reasoning. Include specific technical observations and conclude with a clear determination of the image's authenticity status.>"
            }
 
            Important guidelines:
            1. Focus exclusively on technical indicators of manipulation or AI generation
            2. Be specific about what technical features you observe
            3. Provide evidence-based assessments rather than speculation
            4. Include confidence levels based on clear technical evidence
            5. Return only valid JSON - no additional text or explanations
            """

    response_text = call_gpt4o_with_prompt(image_base64, analysis_prompt)
    
    return response_text

def parse_json_response(response_text):
    """Parse JSON response and handle potential formatting issues"""
    try:
        # Try to parse as-is first
        json_data = json.loads(response_text)
        return json_data, None
    except json.JSONDecodeError:
        try:
            # Try to extract JSON from response if there's extra text
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                json_data = json.loads(json_str)
                return json_data, None
        except json.JSONDecodeError:
            pass
        
        return None, f"Failed to parse JSON response: {response_text[:200]}..."

def process_directory(directory_path):
    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    
    print("Digital Forensics Analysis with GPT-4o")
    print("=" * 70)
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(supported_extensions):
                full_path = os.path.join(root, file)
                print(f"\n Analyzing: {full_path}")
                print("-" * 50)
                
                try:
                    response = analyze_image_modifications(full_path)
                    
                    if response.startswith("Error:"):
                        print(f"❌ {response}")
                        continue
                    
                    # Parse JSON response
                    json_data, error = parse_json_response(response)
                    
                    if error:
                        print(f"❌ JSON Parsing Error: {error}")
                        print(f"Raw response: {response}")
                        continue
                    
                    # Display the analysis
                    if "modification_analysis" in json_data:
                        analysis = json_data["modification_analysis"]
                        print(f"\n🔬 Digital Forensics Analysis:")
                        print(f"   {analysis}")
                    else:
                        print(f"❌ Missing 'modification_analysis' field in response")
                        print(f"Raw JSON: {json_data}")
                        
                except Exception as e:
                    print(f"❌ Error analyzing {full_path}: {str(e)}")
                
                print("\n" + "=" * 70)

def analyze_single_image(image_path):
    """Analyze a single image and return detailed results"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"📸 Analyzing single image: {image_path}")
    print("=" * 70)
    
    try:
        response = analyze_image_modifications(image_path)
        
        if response.startswith("Error:"):
            print(f"❌ {response}")
            return
        
        # Parse JSON response
        json_data, error = parse_json_response(response)
        
        if error:
            print(f"❌ JSON Parsing Error: {error}")
            print(f"Raw response: {response}")
            return
        
        # Display the analysis with better formatting
        if "modification_analysis" in json_data:
            analysis = json_data["modification_analysis"]
            print(f"\n🔬 Digital Forensics Analysis:")
            print(f"{analysis}")
            
            # Also save as JSON file
            output_file = f"{os.path.splitext(image_path)[0]}_analysis.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"\n Analysis saved to: {output_file}")
        else:
            print(f"❌ Missing 'modification_analysis' field in response")
            print(f"Raw JSON: {json_data}")
            
    except Exception as e:
        print(f"❌ Error analyzing {image_path}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python forensics_analysis.py <image_directory>     # Analyze all images in directory")
        print("  python forensics_analysis.py <single_image_path>   # Analyze single image")
        print("\nThis script uses GPT-4o for comprehensive digital forensics analysis.")
    else:
        path = sys.argv[1]
        
        if os.path.isfile(path):
            # Single image analysis
            analyze_single_image(path)
        elif os.path.isdir(path):
            # Directory analysis
            process_directory(path)
        else:
            print(f"❌ Path not found: {path}")