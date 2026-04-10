import os
import json
from .config import logger
from .extraction.extraction_router import ExtractionRouter
from .preprocessing.text_cleaner import TextCleaner
from .chunking.chunker import Chunker
from .classification.document_classifier import DocumentClassifier
from .classification.clause_classifier import ClauseClassifier
from .segmentation.semantic_segmenter import SemanticSegmenter
from .graph.clause_graph_builder import ClauseGraphBuilder
from .validation.json_validator import JSONValidator
from .auth import verify_api_key

def run_advanced_pipeline(file_path, api_key=None):
    """Orchestrates the advanced v3 document intelligence pipeline."""
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

        # Step 3: Document Classification
        doc_classifier = DocumentClassifier()
        doc_info = doc_classifier.classify(clean_text)
        contract_type = doc_info["contract_type"]

        # Step 4: Semantic Segmentation
        segmenter = SemanticSegmenter()
        segmented_clauses = segmenter.segment_text(clean_text)

        # Step 5: Clause Classification
        clause_classifier = ClauseClassifier()
        classified_clauses = clause_classifier.batch_classify(segmented_clauses)

        # Step 6: Build Graph & Chunk
        graph_builder = ClauseGraphBuilder(contract_type)
        graph = graph_builder.build_graph(classified_clauses)
        
        chunker = Chunker()
        final_graph = chunker.chunk_clauses(graph)

        # Step 7: Validation
        validator = JSONValidator()
        is_valid, error = validator.validate_graph(final_graph)
        
        # Step 8: Save Output
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "sample_output_v3.json")
        
        with open(output_file, "w") as f:
            json.dump(final_graph, f, indent=2)

        return {
            "status": "success",
            "contract_type": contract_type,
            "clause_count": len(final_graph),
            "output_file": output_file,
            "graph": final_graph,
            "validation": {"is_valid": is_valid, "error": error}
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    run_advanced_pipeline(args.file)
