import re
import json
from ..utils import client
from ..config import logger

class SemanticSegmenter:
    def __init__(self):
        pass

    def _extract_json_from_response(self, response: str) -> str:
        """Strip any prose/markdown wrapping and extract raw JSON."""
        # Remove markdown code fences
        response = re.sub(r"```json|```", "", response).strip()

        # Find first '[' or '{' and slice from there
        for start_char in ["[", "{"]:
            idx = response.find(start_char)
            if idx != -1:
                response = response[idx:]
                break

        return response

    def _safe_parse(self, response: str, window_num: int):
        """Try to salvage partial or malformed JSON responses."""
        # Attempt 1: Direct parse
        try:
            result = json.loads(response)
            # If model returned a JSON-encoded string, double-parse it
            if isinstance(result, str):
                logger.warning(f"Window {window_num}: Got JSON string, attempting double-parse.")
                result = json.loads(result)
            return result
        except json.JSONDecodeError:
            pass

        # Attempt 2: Trim to last complete clause '}]'
        try:
            trimmed = response[:response.rfind("}]") + 2]
            if trimmed:
                return json.loads(trimmed)
        except Exception:
            pass

        # Attempt 3: Trim to last '}' and wrap as array
        try:
            trimmed = response[:response.rfind("}") + 1]
            if trimmed:
                return json.loads(f"[{trimmed}]")
        except Exception:
            pass

        # Attempt 4: Model forgot array brackets
        try:
            return json.loads(f"[{response}]")
        except Exception:
            pass

        logger.warning(f"Window {window_num}: Could not recover JSON after all parse attempts. Skipping.")
        logger.debug(f"Window {window_num}: Failed response sample: {response[:300]}")
        return None

    def _normalize_to_list(self, data, window_num: int, failed_windows: list):
        """Normalize any model output shape into a flat list of clause dicts."""
        # Already a list
        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            # Case 1: {"clauses": [...]}
            if "clauses" in data and isinstance(data["clauses"], list):
                return data["clauses"]

            # Case 2: Single clause object {"clause_title": "...", "text": "..."}
            if "clause_title" in data and "text" in data:
                logger.info(f"Window {window_num}: Single clause object detected, wrapping as list.")
                return [data]

            # Case 3: Any key containing a list
            list_values = [v for v in data.values() if isinstance(v, list)]
            if list_values:
                logger.info(f"Window {window_num}: Found nested list in dict, extracting.")
                return list_values[0]

            # Case 4: Dict of clause dicts e.g. {"1": {"clause_title": ..., "text": ...}}
            nested = [v for v in data.values() if isinstance(v, dict) and "text" in v]
            if nested:
                logger.info(f"Window {window_num}: Found dict-of-clauses, converting to list.")
                return nested

        logger.warning(f"Window {window_num}: Cannot normalize data of type {type(data)}. Skipping.")
        failed_windows.append(window_num)
        return None

    def segment_text(self, document_text: str):
        """Advanced semantic segmentation for clauses with full JSON recovery."""
        system_prompt = (
            "You are a legal document structure specialist. "
            "Segment the following text into logically complete clauses. "
            "Note that the text contains metadata markers like [PAGE_N] which indicate physical page boundaries. "
            "For each clause, extract: "
            "1. clause_title: The name of the section or heading. "
            "2. text: The full content of the clause. "
            "3. section_number: The numbering prefix (e.g. 1.1, 2.a). "
            "4. page_number: The number 'N' from the most recent [PAGE_N] marker seen before or during the clause text.\n\n"
            "IMPORTANT: Return ONLY a valid, complete JSON array. No explanation, no markdown. "
            "Format: [{\"clause_title\": \"...\", \"text\": \"...\", \"section_number\": \"...\", \"page_number\": 1}]"
        )

        logger.info("Performing advanced semantic segmentation...")

        max_len = 8000
        text_segments = [
            document_text[i:i + max_len]
            for i in range(0, len(document_text), max_len)
        ]

        total_windows = len(text_segments)
        all_clauses = []
        failed_windows = []

        for i, segment in enumerate(text_segments):
            window_num = i + 1
            logger.info(f"Segmenting window {window_num}/{total_windows}...")

            try:
                response = client.get_completion(system_prompt, segment, json_mode=True)

                if not response:
                    logger.warning(f"Window {window_num}: No response after all retries. Skipping.")
                    failed_windows.append(window_num)
                    continue

                # Step 1: Strip prose/markdown
                clean_response = self._extract_json_from_response(response)

                # Step 2: Parse JSON safely
                data = self._safe_parse(clean_response, window_num)
                if data is None:
                    failed_windows.append(window_num)
                    continue

                # Step 3: Normalize whatever shape the model returned into a list
                data = self._normalize_to_list(data, window_num, failed_windows)
                if data is None:
                    continue

                # Step 4: Extract valid clauses
                valid_count = 0
                for c in data:
                    if not isinstance(c, dict):
                        logger.warning(f"Window {window_num}: Skipping non-dict item: {type(c)}")
                        continue
                    if "text" not in c:
                        logger.warning(f"Window {window_num}: Discarding clause missing 'text' field.")
                        continue
                    all_clauses.append({
                        "clause_title": c.get("clause_title", "Untitled Clause"),
                        "text": c["text"],
                        "section_number": c.get("section_number", ""),
                        "page_number": c.get("page_number", 0)
                    })
                    valid_count += 1

                logger.info(f"Window {window_num}: Extracted {valid_count} valid clauses.")

            except Exception as e:
                logger.error(f"Window {window_num}: Unexpected error — {e}. Continuing.")
                failed_windows.append(window_num)
                continue

        logger.info(
            f"Segmentation complete. "
            f"Total clauses: {len(all_clauses)} | "
            f"Windows processed: {total_windows} | "
            f"Failed windows: {failed_windows if failed_windows else 'None'}"
        )

        return all_clauses