import os
import json
import time
import uuid
from typing import Dict, Any, List, Optional
from threading import Lock

# Reuse existing modules
from doc_intelligence.config import logger
from doc_intelligence.extraction.extraction_router import ExtractionRouter
from doc_intelligence.preprocessing.text_cleaner import TextCleaner
from doc_intelligence.classification.document_classifier import DocumentClassifier
from doc_intelligence.classification.clause_classifier import ClauseClassifier
from doc_intelligence.segmentation.semantic_segmenter import SemanticSegmenter
from doc_intelligence.fields.field_extractor import FieldExtractor
from doc_intelligence.graph.clause_graph_builder import ClauseGraphBuilder
from knowledge_layer.main import KnowledgeLayer

class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentOrchestrator:
    """
    Central orchestrator for the 10-step Legal Intelligence Pipeline.
    Manages state, sequences modules, and prepares demo-ready data payloads.
    """
    
    def __init__(self):
        self.jobs = {}
        self.lock = Lock()
        # Use absolute path relative to this file to avoid CWD issues
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.registry_path = os.path.join(base_dir, "jobs.json")
        self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r") as f:
                    self.jobs = json.load(f)
            except:
                self.jobs = {}

    def _save_registry(self):
        with self.lock:
            # Ensure directory exists (though it should as it's the module dir)
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            with open(self.registry_path, "w") as f:
                json.dump(self.jobs, f, indent=4)

    def create_job(self, file_name: str) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "file_name": file_name,
            "status": JobStatus.PENDING,
            "progress": 0,
            "current_step": "Initializing",
            "pipeline_steps": [], # Meticulous diagnostic history
            "created_at": time.time(),
            "updated_at": time.time(),
            "results_path": None,
            "error": None
        }
        self._save_registry()
        return job_id

    def update_job(self, job_id: str, **kwargs):
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)
            self.jobs[job_id]["updated_at"] = time.time()
            self._save_registry()

    def record_step(self, job_id: str, name: str, input_data: Any, output_data: Any, status: str = "completed"):
        """Records a meticulous diagnostic snapshot for the Pipeline Viewer."""
        if job_id in self.jobs:
            # Truncate strings/lists for performance while maintaining demo value
            def truncate(obj):
                if isinstance(obj, str): return (obj[:800] + "...") if len(obj) > 800 else obj
                if isinstance(obj, list): return obj[:10]
                if isinstance(obj, dict): return {k: truncate(v) for k, v in list(obj.items())[:5]}
                return obj

            step_log = {
                "name": name,
                "status": status,
                "input": truncate(input_data),
                "output": truncate(output_data),
                "timestamp": time.time()
            }
            self.jobs[job_id].setdefault("pipeline_steps", []).append(step_log)
            self._save_registry()

    async def run_pipeline(self, job_id: str, file_path: str):
        """Executes the full 10-step pipeline with meticulous deduplication pre-check."""
        from app.core.config import OUTPUT_DIR
        file_name = os.path.basename(file_path).replace("\\", "/").split("/")[-1] # Robust basename
        
        try:
            # METICULOUS DEDUPLICATION: Verify against 50-doc library
            self.update_job(job_id, status=JobStatus.PROCESSING, progress=0, current_step="Checking existing knowledge library")
            
            existing_result = None
            if os.path.exists(OUTPUT_DIR):
                for out_file in os.listdir(OUTPUT_DIR):
                    if out_file.endswith(".json") and out_file != "jobs.json":
                        try:
                            with open(os.path.join(OUTPUT_DIR, out_file), "r") as f:
                                data = json.load(f)
                                # Support Legacy (_metadata.document_name) and New (document_name)
                                doc_meta = data.get("_metadata", {})
                                doc_name = data.get("document_name") or doc_meta.get("document_name")
                                
                                # Case-insensitive meticulously check
                                if doc_name and doc_name.lower() == file_name.lower():
                                    existing_result = data
                                    break
                        except Exception as e:
                            logger.debug(f"Skipping malformed JSON {out_file}: {e}")
                            continue

            if existing_result:
                logger.info(f"Meticulous Skip: Reusing existing analysis for {file_name}")
                # Ensure the cloned result has the new job_id for frontend tracking
                existing_result["id"] = job_id
                
                # If legacy result, normalize root-level fields for Intelligence Canvas
                if "_metadata" in existing_result and "document_name" not in existing_result:
                    existing_result["document_name"] = existing_result["_metadata"].get("document_name", file_name)
                    existing_result["processed_at"] = existing_result["_metadata"].get("processed_date", time.time())

                self.update_job(job_id, progress=100, status=JobStatus.COMPLETED, 
                                current_step="Knowledge library hit - reusing results", 
                                results_path=os.path.join(OUTPUT_DIR, f"{job_id}.json"))
                
                with open(os.path.join(OUTPUT_DIR, f"{job_id}.json"), "w") as f:
                    json.dump(existing_result, f, indent=2)
                return

            # Proceed with normal pipeline
            self.update_job(job_id, status=JobStatus.PROCESSING, progress=0, current_step="Document Ingestion")
            time.sleep(0.5)

            # STEP 2: Extraction
            self.update_job(job_id, progress=10, current_step="Hybrid Extraction Strategy")
            router = ExtractionRouter(file_path)
            
            # Diagnostic Callback for page-by-page demo
            def on_extraction_progress(p, msg):
                self.update_job(job_id, progress=10 + (p * 0.15), current_step=f"Engine: {msg}")

            extracted_data = router.extract_full_document(progress_callback=on_extraction_progress)
            # Meticulous Page Marker Injection: Preserve boundaries for the reasoning engine
            full_text_raw = ""
            for p in extracted_data:
                full_text_raw += f"\n\n[PAGE_{p['page_number']}]\n\n{p['text']}"
            
            self.record_step(job_id, "Text Extraction", f"Processing {file_name}", f"Extracted {len(extracted_data)} pages. Page-mapped text built.")

            # STEP 3: Preprocessing
            self.update_job(job_id, progress=25, current_step="Cleaning & Normalizing Text")
            cleaner = TextCleaner()
            clean_text = cleaner.clean(full_text_raw)
            self.record_step(job_id, "Preprocessing", full_text_raw[:500], clean_text[:500])

            # STEP 4: Classification
            self.update_job(job_id, progress=35, current_step="Taxonomy Alignment (DocType)")
            doc_classifier = DocumentClassifier()
            doc_info = doc_classifier.classify(clean_text)
            contract_type = doc_info["contract_type"]
            self.record_step(job_id, "Document Classification", "Full Document Text", doc_info)

            # STEP 5: Segmentation
            self.update_job(job_id, progress=45, current_step="Meticulous Clause Segmentation")
            segmenter = SemanticSegmenter()
            segmented_clauses = segmenter.segment_text(clean_text)
            self.record_step(job_id, "Clause Segmentation", f"Clean text ({len(clean_text)} chars)", f"Identified {len(segmented_clauses)} segments.")

            # STEP 6: Clause Classification
            self.update_job(job_id, progress=55, current_step="Taxonomy Mapping (Clauses)")
            clause_classifier = ClauseClassifier()
            classified_clauses = clause_classifier.batch_classify(segmented_clauses)
            self.record_step(job_id, "Clause Classification", [c['text'] for c in segmented_clauses[:3]], [c['clause_type'] for c in classified_clauses[:3]])

            # STEP 7: Field Extraction
            self.update_job(job_id, progress=65, current_step="Deep Field Extraction")
            field_extractor = FieldExtractor()
            # Note: FieldExtractor.process_graph expects classified clauses wrapped in a graph
            temp_graph_builder = ClauseGraphBuilder(contract_type)
            temp_graph = temp_graph_builder.build_graph(classified_clauses)
            extraction_results = field_extractor.process_graph(temp_graph, clean_text)
            self.record_step(job_id, "Field Extraction", "Document Context", extraction_results["document_fields"])

            # STEP 8 & 9: Embeddings & Knowledge Layer
            self.update_job(job_id, progress=75, current_step="Vectorization & GraphRAG Indexing")
            from app.services.agent_system import agent_system
            
            knowledge_layer = KnowledgeLayer()
            knowledge_layer.process_results(temp_graph, extraction_results["document_fields"], os.path.basename(file_path), doc_id=job_id)
            
            # Meticulous Sparse Index Sync: Ensure the query engine knows about the new document
            # We inject the document_id into the clause metadata for the agent's filtering logic
            for clause in temp_graph:
                if 'metadata' not in clause: clause['metadata'] = {}
                clause['metadata']['document_id'] = job_id
                clause['metadata']['document_name'] = os.path.basename(file_path)
            
            agent_system.sync_bm25(temp_graph)
            
            self.record_step(job_id, "Embedding Generation", f"Indexing {len(temp_graph)} nodes", "ChromaDB + BM25 Sync Complete")

            # STEP 10: Knowledge Graph
            self.update_job(job_id, progress=90, current_step="Building Hierarchical Graph")
            final_graph = temp_graph
            self.record_step(job_id, "Knowledge Graph Construction", f"{len(final_graph)} nodes", "Hierarchical Map Built")

            # STEP 10: Store Results
            self.update_job(job_id, progress=95, current_step="Finalizing & Persisting Intelligence")
            
            output_dir = "doc_intelligence/output"
            os.makedirs(output_dir, exist_ok=True)
            results_path = os.path.join(output_dir, f"{job_id}.json")
            
            demo_payload = {
                "id": job_id,
                "document_name": os.path.basename(file_path),
                "contract_type": contract_type,
                "metadata": doc_info,
                "document_fields": extraction_results["document_fields"],
                "clause_level_fields": extraction_results["clause_level_fields"],
                "clauses": classified_clauses,
                "graph": final_graph,
                "processed_at": time.time()
            }
            
            with open(results_path, "w") as f:
                json.dump(demo_payload, f, indent=2)

            self.update_job(job_id, status=JobStatus.COMPLETED, progress=100, current_step="Pipeline Finished", results_path=results_path)
            logger.info(f"Pipeline completed for Job {job_id}")

        except Exception as e:
            logger.error(f"Pipeline failed for Job {job_id}: {e}")
            self.update_job(job_id, status=JobStatus.FAILED, error=str(e), current_step="Error Encountered")

orchestrator = DocumentOrchestrator()
