class ConfidenceScorer:
    def __init__(self):
        pass

    def compute(self, segmentation_conf: float, classification_conf: float, extraction_quality: float = 1.0) -> float:
        """
        Computes weighted confidence score.
        Formula: 0.4 * segmentation + 0.3 * classification + 0.3 * extraction_quality
        """
        # Ensure values are floats and between 0-1
        seg = float(segmentation_conf or 0.5) # Default to 0.5 if unavailable
        cla = float(classification_conf or 0.0)
        ext = float(extraction_quality or 1.0)
        
        score = (0.4 * seg) + (0.3 * cla) + (0.3 * ext)
        return round(min(max(score, 0.0), 1.0), 3)
