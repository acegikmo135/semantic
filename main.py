from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
import os

API_KEY = os.environ.get("API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Lightweight model for Render free plan
model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"status": "API running"}


@app.post("/ask")
def ask_question(data: QuestionRequest, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    user_question = data.question

    # Fetch FAQs
    response = supabase.table("faqs").select("*").execute()
    faqs = response.data

    if not faqs:
        return {"message": "No data found"}

    questions = [faq["question"] for faq in faqs]

    # Encode
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    db_embeddings = model.encode(questions, convert_to_tensor=True)

    similarities = util.cos_sim(user_embedding, db_embeddings)[0]

    best_index = similarities.argmax().item()
    best_score = similarities[best_index].item()

    # Threshold
    if best_score < 0.6:
        return {"message": "No similar question found"}

    best_faq = faqs[best_index]

    return {
        "similarity": round(best_score * 100, 2),
        "matched_question": best_faq["question"],
        "answer": best_faq["answer"]
    }        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

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
