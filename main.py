import os
import traceback
import contextlib
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI
import json


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI()


# =========================================================
# ENABLE CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST / RESPONSE MODELS
# =========================================================

class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# =========================================================
# TOOL FUNCTION
# =========================================================

def execute_python_code(code: str) -> dict:
    """
    Execute Python code and capture exact stdout or traceback.
    """

    output_buffer = StringIO()

    try:
        # Capture print output
        with contextlib.redirect_stdout(output_buffer):
            exec(code)

        return {
            "success": True,
            "output": output_buffer.getvalue()
        }

    except Exception:
        # Capture full traceback
        return {
            "success": False,
            "output": traceback.format_exc()
        }


# =========================================================
# AI ERROR ANALYSIS
# =========================================================

def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:

    client = OpenAI(
        api_key=os.environ.get("AIPIPE_API_KEY"),

        # Replace with your AI Pipe endpoint
        base_url="https://aipipe.org/openai/v1"
    )

    prompt = f"""
Analyze this Python code and traceback.

Identify the exact line number(s)
where the error occurred.

CODE:
{code}

TRACEBACK:
{traceback_text}

Return JSON ONLY in this format:
{{"error_lines":[3]}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )

    content = response.choices[0].message.content

    parsed = json.loads(content)

    return parsed["error_lines"]

# =========================================================
# API ENDPOINT
# =========================================================

@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):

    # Step 1: Execute Python code
    execution_result = execute_python_code(request.code)

    # Step 2: If successful, return output directly
    if execution_result["success"]:
        return CodeResponse(
            error=[],
            result=execution_result["output"]
        )

    # Step 3: Use AI only if error occurred
    error_lines = analyze_error_with_ai(
        request.code,
        execution_result["output"]
    )

    # Step 4: Return traceback + line numbers
    return CodeResponse(
        error=error_lines,
        result=execution_result["output"]
    )


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():
    return {
        "message": "Code Interpreter API is running"
    }
