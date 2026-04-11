import os
import json
from .config import logger
from .extraction.extraction_router import ExtractionRouter
from .preprocessing.text_cleaner import TextCleaner
from .chunking.chunker import Chunker
from .classification.document_classifier import DocumentClassifier
from .classification.clause_classifier import ClauseClassifier
from .segmentation.semantic_segmenter import SemanticSegmenter
from .refinement.orchestrator import RefinementOrchestrator
from .graph.clause_graph_builder import ClauseGraphBuilder
from .validation.json_validator import JSONValidator
from .fields.field_extractor import FieldExtractor
from .auth import verify_api_key
from knowledge_layer.main import KnowledgeLayer
from knowledge_layer.embeddings.chroma_store import chroma_store

def reset_pipeline_data():
    """Wipes existing data (keeping schema) for a fresh start."""
    logger.warning("RESET TRIGGERED: Clearing existing database and outputs...")
    
    # 1. Reset ChromaDB
    try:
        chroma_store.reset_collection()
    except Exception as e:
        logger.error(f"Failed to reset ChromaDB: {e}")

    # 2. Clear Output Directories
    output_dirs = [
        os.path.join(os.path.dirname(__file__), "output"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_layer", "output")
    ]
    
    for d in output_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".json") or f.endswith(".csv"):
                    try:
                        os.remove(os.path.join(d, f))
                        logger.info(f"Deleted stale file: {f}")
                    except Exception as e:
                        logger.error(f"Failed to delete {f}: {e}")
    
    logger.info("Pipeline reset complete. Starting fresh.")



def run_advanced_pipeline(file_path, api_key=None):
    """Orchestrates the advanced v3 document intelligence pipeline with STRICT JSON mapping."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return {"error": "File not found"}

    # Step 0: Auth
    if api_key and not verify_api_key(api_key):
        return {"error": "Unauthorized"}

    try:
        # Step 1: Hybrid Extraction
        router = ExtractionRouter(file_path)
        extracted_data = router.extract_full_document()
        full_text_raw = "\n\n".join([p["text"] for p in extracted_data])

        # Step 2: Preprocessing
        cleaner = TextCleaner()
        clean_text = cleaner.clean(full_text_raw)
        
        if not clean_text.strip():
            logger.error("Extraction resulted in empty text.")
            return {"error": "Extraction failed (empty text)."}

        # Step 3: Document Classification
        doc_classifier = DocumentClassifier()
        doc_info = doc_classifier.classify(clean_text)
        contract_type = doc_info["contract_type"]

        # Step 4: Semantic Segmentation
        segmenter = SemanticSegmenter()
        segmented_clauses = segmenter.segment_text(clean_text)

        # Step 5: Clause Classification & Intelligence
        clause_classifier = ClauseClassifier()
        classified_clauses = clause_classifier.batch_classify(segmented_clauses)

        # Step 5.5: Refinement Layer
        refiner = RefinementOrchestrator()
        refined_clauses = refiner.refine_document(classified_clauses, clean_text)

        # Step 6: Build Graph & Chunk
        graph_builder = ClauseGraphBuilder(contract_type)
        graph = graph_builder.build_graph(refined_clauses)
        
        chunker = Chunker()
        final_graph = chunker.chunk_clauses(graph)

        # Step 7: Field Extraction
        field_extractor = FieldExtractor()
        extraction_results = field_extractor.process_graph(final_graph, clean_text)

        # Step 7: Validation
        validator = JSONValidator()
        is_valid, error = validator.validate_graph(final_graph)

        # Step 8: Save Output (Dynamic + STRICT JSON)
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        file_basename = os.path.basename(file_path)
        json_filename = os.path.splitext(file_basename)[0] + ".json"
        output_file = os.path.join(output_dir, json_filename)
        
        # Build the document payload (without global graph data first)
        strict_payload = {
            "contract_type": contract_type,
            "document_fields": extraction_results["document_fields"],
            "clauses": [
                {
                    "clause_title": c.get("clause_title", "Unnamed Clause"),
                    "summary": c.get("summary", "Not Found"),
                    "key_points": c.get("key_points", ["Not Found"])
                }
                for c in final_graph
            ] if final_graph else [],
            "_metadata": {
                "status": "processing",
                "document_name": file_basename,
                "clause_count": len(final_graph),
                "processed_date": "2024-04-11",
                "is_valid": is_valid
            }
        }
        
        # INITIAL SAVE (Fail-safe)
        with open(output_file, "w") as f:
            json.dump(strict_payload, f, indent=2)
        logger.info(f"Initial document results saved to {output_file}")

        # Step 10: Knowledge Layer (Vectorization + Global Graph Merge)
        try:
            knowledge_layer = KnowledgeLayer()
            knowledge_layer.process_results(final_graph, extraction_results["document_fields"])
            
            # Update metadata to success
            strict_payload["_metadata"]["status"] = "success"
            with open(output_file, "w") as f:
                json.dump(strict_payload, f, indent=2)
                
        except Exception as e:
            logger.error(f"Knowledge Layer failed for {file_basename}: {e}")
            strict_payload["_metadata"]["status"] = "partial_success_no_graph"
            with open(output_file, "w") as f:
                json.dump(strict_payload, f, indent=2)

        return strict_payload

    except Exception as e:
        err_msg = str(e)
        if "402" in err_msg or "credits" in err_msg.lower():
            logger.error("CRITICAL: API credits exhausted (402). Stopping batch.")
            raise e # Reraise to halt batch execution
        logger.error(f"Pipeline failed for {file_path}: {e}")
        return {"error": str(e)}

def run_batch_pipeline(directory_path, api_key=None):
    """Processes PDF files with skip-if-exists logic and fail-safe halts."""
    import gc
    if not os.path.isdir(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return {"error": "Directory not found"}

    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]
    logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
    
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    batch_results = []
    for pdf in pdf_files:
        json_name = os.path.splitext(pdf)[0] + ".json"
        if os.path.exists(os.path.join(output_dir, json_name)):
            logger.info(f"Skipping {pdf} (Already processed)")
            continue

        logger.info(f"--- Processing: {pdf} ---")
        file_path = os.path.join(directory_path, pdf)
        try:
            res = run_advanced_pipeline(file_path, api_key)
            batch_results.append({
                "file": pdf,
                "status": "success" if "error" not in res else "failed"
            })
        except Exception as e:
            if "402" in str(e) or "credits" in str(e).lower():
                logger.error("Halt: Missing Credits. Please top up your OpenRouter account.")
                break
            batch_results.append({
                "file": pdf,
                "status": "failed",
                "error": str(e)
            })
        finally:
            # Force cleanup of memory after each document
            gc.collect()

    return {
        "status": "batch_completed",
        "total": len(pdf_files),
        "processed": len(batch_results),
        "results": batch_results
    }



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to a single PDF file")
    parser.add_argument("--dir", help="Path to a directory containing PDF files")
    parser.add_argument("--reset", action="store_true", help="Clear existing data before starting")
    args = parser.parse_args()
    
    if args.reset:
        reset_pipeline_data()

    if args.dir:
        run_batch_pipeline(args.dir)
    elif args.file:
        run_advanced_pipeline(args.file)
    else:
        print("Please provide --file or --dir")

