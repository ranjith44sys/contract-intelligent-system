from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
import os
import json
import shutil
import uuid
from app.pipeline.orchestrator import orchestrator, JobStatus
from app.services.agent_system import agent_system
from app.core.config import UPLOADS_DIR, OUTPUT_DIR
from doc_intelligence.config import logger

router = APIRouter()

@router.get("/documents")
def list_intelligent_documents_alias():
    """Alias for /pipeline/documents to prevent 404s."""
    return list_intelligent_documents()

# --- Models ---

class QueryRequest(BaseModel):
    query: str
    selected_ids: list[str] = []

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

# --- Pipeline Routes ---

@router.post("/pipeline/upload")
async def upload_documents(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    """Uploads multiple documents, each preserving its meticulously sanitized original filename."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    
    responses = []
    
    for file in files:
        # Meticulous Name Preservation
        base_name = os.path.splitext(file.filename)[0]
        ext = os.path.splitext(file.filename)[1]
        
        # Sanitize
        import re
        clean_name = re.sub(r'[^\w\s-]', '', base_name).strip().replace(' ', '_')
        if not clean_name: clean_name = "document"
        
        # Final path
        file_path = os.path.join(UPLOADS_DIR, f"{clean_name}{ext}")
        
        # Handle collisions
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(UPLOADS_DIR, f"{clean_name}_{counter}{ext}")
            counter += 1

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        job_id = orchestrator.create_job(file.filename)
        background_tasks.add_task(orchestrator.run_pipeline, job_id, file_path)
        
        responses.append({
            "job_id": job_id, 
            "file_name": file.filename,
            "message": f"Pipeline started for {os.path.basename(file_path)}"
        })
    
    return responses

@router.get("/pipeline/jobs/{job_id}/status")
def get_job_status(job_id: str):
    """Polls for job status and progress."""
    if job_id not in orchestrator.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return orchestrator.jobs[job_id]

@router.get("/pipeline/documents")
def list_intelligent_documents():
    """Lists all successfully processed documents from the 50-file library."""
    if not os.path.exists(OUTPUT_DIR):
        return []
    
    docs = []
    # Track seen names to avoid duplicates in the UI
    seen_names = set()
    
    # Sort files by time for stable order
    files = sorted(os.listdir(OUTPUT_DIR), key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
    
    for file in files:
        if file.endswith(".json") and not file == "jobs.json":
            file_path = os.path.join(OUTPUT_DIR, file)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    
                    # Normalization for legacy/new formats
                    meta = data.get("_metadata", {})
                    name = data.get("document_name") or meta.get("document_name") or file
                    
                    if name in seen_names: continue
                    seen_names.add(name)

                    docs.append({
                        "id": data.get("id", file.replace(".json", "")),
                        "name": name,
                        "type": data.get("contract_type", "Contract"),
                        "processed_at": data.get("processed_at") or meta.get("processed_date")
                    })
            except:
                continue
    return docs

@router.get("/pipeline/documents/{doc_id}/analysis")
def get_analysis_results(doc_id: str):
    """Returns the full intelligence payload for the dashboard."""
    file_path = os.path.join(OUTPUT_DIR, f"{doc_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Analysis result not found")
    
    with open(file_path, "r") as f:
        data = json.load(f)
        # Ensure ID is present for the knowledge graph fetch
        if "id" not in data:
            data["id"] = doc_id
        return data

# --- Agentic RAG Routes ---

@router.post("/pipeline/query")
def agentic_query(data: QueryRequest):
    """Executes the 6-step Agentic RAG flow with strict source filtering."""
    try:
        results = agent_system.run(data.query, selected_ids=data.selected_ids)
        return results
    except Exception as e:
        logger.error(f"RAG Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline/documents/{doc_id}/graph")
async def get_document_graph(doc_id: str):
    """Returns a meticulous 3-layer hierarchical knowledge graph for visualization."""
    file_path = os.path.join(OUTPUT_DIR, f"{doc_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document not found")
    
    with open(file_path, "r") as f:
        data = json.load(f)
        
    clauses = data.get("clauses", [])
    doc_fields = data.get("document_fields", {})
    clause_fields = data.get("clause_level_fields", [])
    
    nodes = []
    links = []
    
    # --- LAYER 0: ROOT DOCUMENT ---
    root_id = "root"
    nodes.append({
        "id": root_id,
        "name": data.get("document_name", "Document"),
        "group": "document",
        "type": data.get("contract_type", "Legal Content")
    })
    
    # --- LAYER 1: GLOBAL DOCUMENT FIELDS ---
    # Connect high-level metadata directly to root
    for key, val in doc_fields.items():
        if isinstance(val, (str, int, float)) and val != "Not Found":
            field_id = f"doc_f_{key}"
            nodes.append({
                "id": field_id,
                "name": f"{key.replace('_', ' ').capitalize()}: {val}",
                "group": "doc_field",
                "parent": root_id
            })
            links.append({"source": root_id, "target": field_id, "label": "contains"})
        elif key == "parties" and isinstance(val, list):
            for i, party in enumerate(val):
                p_id = f"party_{i}"
                nodes.append({
                    "id": p_id,
                    "name": party.get("name", "Unknown Party"),
                    "group": "party",
                    "sub_label": party.get("role", "Party")
                })
                links.append({"source": root_id, "target": p_id, "label": "involves"})

    # --- LAYER 2: CLAUSES & THEIR SEMANTIC CHILDREN ---
    for i, c in enumerate(clauses):
        cid = c.get("clause_id", f"cl_{i}")
        nodes.append({
            "id": cid,
            "name": c.get("clause_title", "General Provision"),
            "group": "clause",
            "type": c.get("clause_type", "Standard")
        })
        links.append({"source": root_id, "target": cid, "label": "segment"})
        
        # Link specialized clause fields (Layer 3)
        # Find fields belonging to this clause id
        associated_field_set = []
        if isinstance(clause_fields, list):
            for cf in clause_fields:
                if cf.get("clause_id") == cid:
                    extracted = cf.get("extracted_fields", {})
                    for field_key, field_val in extracted.items():
                        if field_val != "Not Found":
                            cf_id = f"cf_{cid}_{field_key}"
                            nodes.append({
                                "id": cf_id,
                                "name": f"{field_key.replace('_', ' ').capitalize()}",
                                "group": "field_value",
                                "val": str(field_val)
                            })
                            links.append({"source": cid, "target": cf_id, "label": "defines"})

    return {"nodes": nodes, "links": links}


