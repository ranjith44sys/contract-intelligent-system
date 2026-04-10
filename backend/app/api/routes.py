from fastapi import APIRouter
from pydantic import BaseModel
from app.agent.translator import translate_query

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
