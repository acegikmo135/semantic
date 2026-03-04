import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Load model ONCE at startup (important for performance)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Get API key from environment variable
API_KEY = os.getenv("API_KEY")


# Request body model
class SimilarityRequest(BaseModel):
    question1: str
    question2: str


def calculate_similarity(q1, q2):
    embeddings = model.encode([q1, q2], normalize_embeddings=True)
    similarity = np.dot(embeddings[0], embeddings[1])
    return float(similarity * 100)


# Secure endpoint
@app.post("/v1/similarity")
def similarity_api(
    request: SimilarityRequest,
    x_api_key: str = Header(None)
):
    # Check API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

    score = calculate_similarity(request.question1, request.question2)

    if score > 80:
        result = "Very Similar"
    elif score > 50:
        result = "Moderately Similar"
    else:
        result = "Not Similar"

    return {
        "similarity": round(score, 2),
        "result": result
    }
