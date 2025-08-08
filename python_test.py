from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime
import os
import traceback
import json
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Supabase setup
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

app = FastAPI()

# ✅ CORS configuration for frontend served from Live Server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/lessons")
def get_lessons():
    try:
        response = supabase.table("lessons").select("id, title").execute()
        return response.data
    except Exception as e:
        print("Error loading lessons:", e)
        raise HTTPException(status_code=500, detail="Could not load lessons.")

@app.get("/lessons/{lesson_id}")
def get_lesson_detail(lesson_id: str):
    try:
        response = supabase.table("lessons").select("*").eq("id", lesson_id).single().execute()
        lesson = response.data
        if not lesson:
            raise HTTPException(status_code=404, detail="Lesson not found")
        return lesson
    except Exception as e:
        print("Error fetching lesson:", e)
        raise HTTPException(status_code=500, detail="Could not fetch lesson")

class ChatRequest(BaseModel):
    lesson_id: str
    user_input: str
    chat_history: list  # List of {"user": "...", "assistant": "..."}

@app.post("/chat")
def chat(chat_req: ChatRequest):
    try:
        lesson_resp = supabase.table("lessons").select("*").eq("id", chat_req.lesson_id).single().execute()
        lesson = lesson_resp.data
        if not lesson:
            raise HTTPException(status_code=404, detail="Lesson not found")

        system_prompt = lesson["system_prompt"]

        rubric_resp = supabase.table("rubrics").select("*").eq("lesson_id", chat_req.lesson_id).execute()
        rubrics = rubric_resp.data

        messages = [{"role": "system", "content": system_prompt}]
        for pair in chat_req.chat_history:
            messages.append({"role": "user", "content": pair["user"]})
            messages.append({"role": "assistant", "content": pair["assistant"]})
        messages.append({"role": "user", "content": chat_req.user_input})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )
        assistant_reply = response.choices[0].message.content

        scores = {r["criterion"]: r["max_score"] for r in rubrics}  # placeholder

        return {"response": assistant_reply, "scores": scores}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

class ScoreRequest(BaseModel):
    lesson_id: str
    chat_history: list  # List of {"user": "...", "assistant": "..."}

@app.post("/score")
def score_conversation(req: ScoreRequest):
    try:
        # Fetch lesson and evaluation prompt
        lesson_resp = supabase.table("lessons").select("*").eq("id", req.lesson_id).single().execute()
        lesson = lesson_resp.data
        if not lesson:
            raise HTTPException(status_code=404, detail="Lesson not found")

        system_prompt = lesson.get("system_prompt", "")
        evaluation_prompt = lesson.get("evaluation_prompt", "")

        # Get rubric criteria
        rubric_resp = supabase.table("rubrics").select("*").eq("lesson_id", req.lesson_id).execute()
        rubrics = rubric_resp.data

        # Build chat transcript
        chat_text = "\n".join([f"User: {p['user']}\nAssistant: {p['assistant']}" for p in req.chat_history])
        rubric_str = "\n".join([f"- {r['criterion']} (1 to {r['max_score']})" for r in rubrics])

        llm_prompt = (
            f"{evaluation_prompt.strip()}\n\n"
            f"Conversation:\n{chat_text}\n\n"
        )
        # Call LLM
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.2
        )

        # Parse JSON result from LLM
        try:
            raw_content = response.choices[0].message.content.strip()
            scores = json.loads(raw_content)
        except Exception:
            traceback.print_exc()
            scores = {
                r["criterion"]: {"score": None, "feedback": ""}
                for r in rubrics
            }
            scores["suggestions"] = []
            scores["strength"] = ""

        # Store in Supabase
        supabase.table("chat_logs").insert({
            "lesson_id": req.lesson_id,
            "chat_history": req.chat_history,
            "scores": scores,
            "timestamp": datetime.utcnow().isoformat(),
            "system_prompt": system_prompt,
            "evaluation_prompt": evaluation_prompt
        }).execute()

        # Return full info
        return {
            "scores": scores,
            "rubric": rubrics,
            "suggestions": scores.get("suggestions", []),
            "strength": scores.get("strength", "")
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")