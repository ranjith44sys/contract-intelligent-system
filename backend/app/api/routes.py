from fastapi import APIRouter
from pydantic import BaseModel
from app.agent.translator import translate_query
from doc_intelligence.main import run_advanced_pipeline

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

class QueryTranslationResponse(BaseModel):
    original: str
    expanded: str
    rewritten: str
    decomposed: list[str]

@router.post("/translate", response_model=QueryTranslationResponse)
def translate(data: QueryRequest):
    result = translate_query(data.query)
    return result

class ExtractRequest(BaseModel):
    file_path: str
    api_key: str

@router.post("/extract")
def extract(data: ExtractRequest):
    result = run_advanced_pipeline(data.file_path, api_key=data.api_key)
    return result
