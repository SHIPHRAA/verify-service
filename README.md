Create requirements.txt file for faster builds

```shell
# Install poetry's export plugin
poetry self add poetry-plugin-export

# Create the requirements file
poetry export -f requirements.txt --output requirements.txt --without-hashes

# Only CPU
# Generate requirements.txt with only main + ml-cpu dependencies
poetry export --only=main,ml-cpu -f requirements.txt --output requirements.txt --without-hashes
```

---

## Usage

### 1. Test with Verify Service (FastAPI)

Run the backend:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then analyze all images in a directory:

```bash
python test_verify.py static/images/
```

**Output**: Pretty JSON with face-based and object-based fake percentages.

---

### 2. Test with GPT-4o (OpenAI)

Create a `.env` file at the root with your API key:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Install runtime dependencies:

```bash
pip install openai python-dotenv
```

Analyze the same dataset using GPT-4o's vision model:

```bash
python gpt4o_image_analysis.py static/images/
```

**Output**: GPT-4o returns a single-line estimate like:

```
AI-generated: 90%
Real image: 8% AI-likelihood
```

Retries are built-in if GPT-4o hesitates.

---

### 3. Compare Verify Service vs GPT-4o

Run the unified comparison script:

```bash
python compare_verify_and_gpt4o.py static/images/
```

**Output Format (per image)**:

```
Image: static/images/fake/m0201.jpeg
Image ID: 77f63ad2-...

Verify Service:
{
  "fake_percentage": 0.48,
  "bounding_boxes": [...],
  ...
}

GPT-4o:
AI-generated: 90%
```

