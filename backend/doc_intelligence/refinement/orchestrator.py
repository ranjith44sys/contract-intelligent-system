import concurrent.futures
from .text_normalizer import TextNormalizer
from .title_refiner import TitleRefiner
from .clause_splitter import ClauseSplitter
from .type_normalizer import TypeNormalizer
from .definition_extractor import DefinitionExtractor
from .reference_extractor import ReferenceExtractor
from .confidence_scorer import ConfidenceScorer
from ..config import logger

class RefinementOrchestrator:
    def __init__(self):
        self.text_normalizer = TextNormalizer()
        self.title_refiner = TitleRefiner()
        self.clause_splitter = ClauseSplitter()
        self.type_normalizer = TypeNormalizer()
        self.definition_extractor = DefinitionExtractor()
        self.reference_extractor = ReferenceExtractor()
        self.confidence_scorer = ConfidenceScorer()

    def refine_document(self, clauses: list, full_context: str) -> list:
        """
        Runs the refinement process for all clauses in parallel.
        """
        logger.info(f"Starting parallel refinement for {len(clauses)} clauses...")
        
        refined_clauses = []
        
        # We process in two stages:
        # 1. Split overloaded clauses (Sequential as it changes the list length)
        stage1_clauses = []
        for clause in clauses:
            split_results = self.clause_splitter.split_if_needed(clause)
            stage1_clauses.extend(split_results)

        # 2. Refine each clause (Parallel)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_clause = {
                executor.submit(self._refine_single_clause, clause, full_context): clause 
                for clause in stage1_clauses
            }
            for future in concurrent.futures.as_completed(future_to_clause):
                try:
                    refined = future.result()
                    if refined:
                        refined_clauses.append(refined)
                except Exception as e:
                    logger.error(f"Error refining clause: {e}")

        # Re-sort by page number and original order
        refined_clauses.sort(key=lambda x: (x.get("page_number", 0), x.get("section_number", "")))
        return refined_clauses

    def _refine_single_clause(self, clause: dict, full_context: str) -> dict:
        """Helper to refine one clause."""
        try:
            # 1. Normalize Text
            clause["raw_text"] = clause.get("text", "") # Keep original
            clause["text"] = self.text_normalizer.normalize(clause["text"])
            
            # 2. Refine Title
            clause["clause_title"] = self.title_refiner.refine(clause["text"], full_context)
            
            # 3. Normalize Type
            type_info = self.type_normalizer.normalize_type(clause.get("clause_type", "Other"), clause["text"])
            clause["clause_type"] = type_info["clause_type"]
            classification_conf = type_info["confidence"]
            
            # 4. Extract Definitions
            clause["definitions"] = self.definition_extractor.extract(clause["text"])
            
            # 5. Extract References
            clause["references"] = self.reference_extractor.extract(clause["text"])
            
            # 6. Final Confidence Score
            # Assuming segmentation confidence is 0.9 if not provided
            clause["confidence"] = self.confidence_scorer.compute(
                segmentation_conf=0.9, 
                classification_conf=classification_conf
            )
            
            return clause
        except Exception as e:
            logger.error(f"Failed to refine individual clause: {e}")
            return clause
