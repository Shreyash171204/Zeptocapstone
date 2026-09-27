import json
import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"},
)


def load_documents():
    documents = []
    ids = []
    metadatas = []

    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()

        if text:
            documents.append(text)
            ids.append(path.stem)
            metadatas.append({"source": path.name})

    return ids, documents, metadatas


def build_index():
    ids, documents, metadatas = load_documents()

    if not documents:
        raise RuntimeError("No policy documents found.")

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Indexed {len(documents)} policy documents.")


build_index()


def retrieve(query: str, top_k: int = 3):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    results = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        results.append(
            {
                "document": document,
                "source": metadata.get("source", "unknown"),
                "distance": float(distance),
            }
        )

    return results


PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer-support assistant.

CONTEXT:
Use only the retrieved Zepto policy context provided below.

TASK:
Answer the customer's question accurately using the policy context.

FORMAT:
Return JSON with:
{
  "answer": "...",
  "sources": ["..."],
  "confidence": 0.0
}

LENGTH:
Keep the answer concise and customer-friendly.

NEGATIVE CONSTRAINT:
Do not invent policies, refunds, delivery times, eligibility rules, or other facts
that are not supported by the provided context.

FEW-SHOT EXAMPLE:
Question: How long does delivery usually take?
Answer:
{
  "answer": "Delivery times depend on the applicable delivery policy.",
  "sources": ["doc_01.txt"],
  "confidence": 0.90
}

RETRIEVED CONTEXT:
{context}

CUSTOMER QUESTION:
{question}
"""


def build_prompt(question: str, retrieved):
    context_parts = []

    for item in retrieved:
        context_parts.append(
            f"[{item['source']}]\n{item['document']}"
        )

    context = "\n\n".join(context_parts)

    return PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )


def classify_intent(query: str) -> str:
    q = query.lower()

    policy_keywords = [
        "delivery",
        "deliver",
        "return",
        "refund",
        "cancel",
        "cancellation",
        "membership",
        "tracking",
        "track",
        "gift card",
        "support",
        "hours",
        "replacement",
        "payment",
    ]

    if any(keyword in q for keyword in policy_keywords):
        return "policy_question"

    return "general_question"


def mock_llm(question: str, retrieved):
    if not retrieved:
        return {
            "answer": "I could not find relevant policy information.",
            "sources": [],
            "confidence": 0.20,
        }

    top = retrieved[0]

    answer = (
        "Based on the available Zepto policy information: "
        + top["document"][:500]
    )

    sources = [item["source"] for item in retrieved]

    return {
        "answer": answer,
        "sources": sources,
        "confidence": 0.85,
    }


def generate_answer(question: str, retrieved):
    prompt = build_prompt(question, retrieved)

    # Offline baseline required by the assignment.
    if os.getenv("MOCK_LLM", "1") == "1":
        result = mock_llm(question, retrieved)
        return result

    # Optional real LLM path.
    # The application can still run in MOCK_LLM=1 mode without this package.
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise RuntimeError(
            "Install langchain-groq to use MOCK_LLM=0."
        )

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required when MOCK_LLM=0."
        )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0,
    )

    last_error = None

    # Initial attempt + 2 retries.
    for _ in range(3):
        try:
            response = llm.invoke(prompt)

            content = response.content

            parsed = json.loads(content)

            return {
                "answer": str(parsed["answer"]),
                "sources": list(parsed["sources"]),
                "confidence": float(parsed["confidence"]),
            }

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"LLM failed after 3 attempts: {last_error}"
    )