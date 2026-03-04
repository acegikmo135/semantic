import os
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ===============================
# Environment Variables
# ===============================

SECRET_KEY = os.getenv("API_SECRET_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SECRET_KEY:
    raise ValueError("API_SECRET_KEY not set")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials not set")

# ===============================
# Initialize App
# ===============================

app = FastAPI()

# Load embedding model (CPU friendly)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===============================
# Request Model
# ===============================

class QuestionRequest(BaseModel):
    question: str

# ===============================
# API Key Verification
# ===============================

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != SECRET_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key"
        )

# ===============================
# Similarity Function
# ===============================

def calculate_similarity(q1: str, q2: str) -> float:
    embeddings = model.encode([q1, q2])
    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]
    return similarity * 100

# ===============================
# API Endpoint
# ===============================

@app.post("/ask", dependencies=[Depends(verify_api_key)])
def ask_question(data: QuestionRequest):

    user_question = data.question

    # Fetch stored questions from Supabase
    response = supabase.table("questions").select("*").execute()

    if not response.data:
        raise HTTPException(
            status_code=404,
            detail="No questions found in database"
        )

    best_match = None
    highest_score = 0

    for row in response.data:
        stored_question = row["question"]
        stored_answer = row["answer"]

        score = calculate_similarity(user_question, stored_question)

        if score > highest_score:
            highest_score = score
            best_match = {
                "question": stored_question,
                "answer": stored_answer,
                "similarity": round(score, 2)
            }

    if highest_score < 50:
        return {
            "similarity": round(highest_score, 2),
            "result": "No similar question found"
        }

    return {
        "similarity": best_match["similarity"],
        "matched_question": best_match["question"],
        "answer": best_match["answer"]
    }

# ===============================
# Health Check
# ===============================

@app.get("/")
def root():
    return {"status": "API is running"}
