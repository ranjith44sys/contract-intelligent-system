from .taxonomy import CONTRACT_TYPES, CLAUSE_TYPES

def normalize_contract_type(label: str) -> str:
    """Maps fuzzy document labels to canonical taxonomy labels."""
    label_map = {
        "msa": "Master Service Agreement (MSA)",
        "master service agreement": "Master Service Agreement (MSA)",
        "nda": "Non-Disclosure Agreement (NDA)",
        "non-disclosure agreement": "Non-Disclosure Agreement (NDA)",
        "sla": "Service Level Agreement (SLA)",
        "service level agreement": "Service Level Agreement (SLA)",
        "employment agreement": "Employment Contract",
        "partnership": "Partnership Agreement",
        "vendor": "Vendor Agreement"
    }
    
    clean_label = label.lower().strip()
    
    # Direct check
    for canonical in CONTRACT_TYPES:
        if clean_label == canonical.lower():
            return canonical
            
    # Fuzzy/Common mapping
    for key, val in label_map.items():
        if key in clean_label:
            return val
            
    return "Others"

def normalize_clause_type(label: str) -> str:
    """Maps fuzzy clause labels to canonical taxonomy labels."""
    clean_label = label.lower().strip().replace(" clause", "").replace(" provision", "")
    
    for canonical in CLAUSE_TYPES:
        if clean_label == canonical.lower():
            return canonical
            
    return "Others"
