import os
import re
try:
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


def fallback_translate(query: str) -> dict:
    """Fallback rule-based translator when OpenAI/Langchain isn't setup."""
    query_lower = query.lower().strip()
    synonyms_map = {
        "liability": ["liability", "indemnity", "responsibility"],
        "fraud": ["fraud", "misrepresentation", "deceit"],
        "termination": ["termination", "cancellation", "end of contract"],
        "audit": ["audit", "inspection", "review"],
        "breach": ["breach", "violation", "infringement"],
        "risk": ["risk", "danger", "hazard"],
        "contract": ["contract", "agreement"]
    }
    
    expanded_terms = []
    words = re.findall(r'\b\w+\b', query_lower)
    for word in words:
        if word in synonyms_map:
            expanded_terms.extend(synonyms_map[word])
    if not expanded_terms:
        expanded_str = query
    else:
        unique_expanded = list(dict.fromkeys(expanded_terms))
        expanded_str = " OR ".join(unique_expanded)
        if "fraud exception" in query_lower and "fraud exception" not in expanded_str:
            expanded_str += " OR fraud exception"
    if query_lower == "fraud liability":
        expanded_str = "liability OR indemnity OR fraud exception"

    rewritten_str = query
    if "fraud liability" in query_lower:
        rewritten_str = "What are the liability clauses related to fraud?"
    elif "termination" in query_lower:
        rewritten_str = "What are the conditions for contract termination?"
    elif "audit" in query_lower:
        rewritten_str = "What are the audit rights and obligations defined in the contract?"
    elif len(words) > 0:
        rewritten_str = f"What are the clauses regarding {query_lower}?"

    decomposed = []
    if "fraud liability" in query_lower:
        decomposed = ["liability clause", "fraud condition"]
    elif "risk" in query_lower:
        decomposed = ["liability clause", "termination clause", "audit rights"]
    else:
        if len(words) > 1:
            decomposed = [f"{w} clause" for w in words if len(w) > 3][:3]
            if not decomposed:
                decomposed = [query_lower]
        else:
            decomposed = [f"{query_lower} related conditions"]

    return {
        "original": query,
        "expanded": expanded_str,
        "rewritten": rewritten_str,
        "decomposed": decomposed
    }


def translate_query(query: str) -> dict:
    """
    Transforms the user query into structured, retrieval-friendly forms using LLMs (if OPENAI_API_KEY is available),
    or falls back to rules if it isn't set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not LANGCHAIN_AVAILABLE or not api_key:
        print("Langchain/OpenAI API key missing. Falling back to rule-based dictionary.")
        return fallback_translate(query)

    try:
        llm = ChatOpenAI(temperature=0)
        
        # 1. Multi-Query Expansion
        prompt_perspectives = ChatPromptTemplate.from_template(
            "You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. Provide these alternative questions separated by newlines. Original question: {question}"
        )
        generate_queries_chain = prompt_perspectives | llm | StrOutputParser()
        expanded_queries_raw = generate_queries_chain.invoke({"question": query})
        expanded_list = [q.strip() for q in expanded_queries_raw.split('\n') if q.strip()]
        # Create a boolean OR-like string representing expansion logic
        expanded_str = " OR ".join(expanded_list[:3])  # keep it to 3 to simulate synonyms expansion
        
        # 2. Query Rewriting (RAG Fusion inspired)
        prompt_rewrite = ChatPromptTemplate.from_template(
            "You are a legal contract assistant. Rewrite the following vague query into a clear, full grammatical sentence explicitly describing what clauses or conditions the user might be looking for. \nQuery: {question}\nRewritten Full Sentence:"
        )
        rewrite_chain = prompt_rewrite | llm | StrOutputParser()
        rewritten_str = rewrite_chain.invoke({"question": query}).strip()

        # 3. Query Decomposition
        prompt_decomposition = ChatPromptTemplate.from_template(
            "You are a helpful assistant that generates multiple sub-questions or sub-topics related to an input question.\nThe goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation.\nGenerate multiple search queries related to: {question}\nOutput exactly 3 queries, separated by newlines, with NO bullet points or numbering format:"
        )
        decomposition_chain = prompt_decomposition | llm | StrOutputParser()
        decomposed_raw = decomposition_chain.invoke({"question": query})
        decomposed = [q.strip('- 1234567890.') for q in decomposed_raw.split('\n') if q.strip()]

        return {
            "original": query,
            "expanded": expanded_str,
            "rewritten": rewritten_str,
            "decomposed": decomposed[:3]
        }

    except Exception as e:
        print(f"LLM translation failed: {e}. Falling back to rules.")
        return fallback_translate(query)
