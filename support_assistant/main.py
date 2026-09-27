from typing import TypedDict

from fastapi import FastAPI
from langgraph.graph import StateGraph, END

from models import AskRequest, AnswerResponse
from rag import classify_intent, retrieve, generate_answer


app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0.0",
)


class GraphState(TypedDict, total=False):
    query: str
    intent: str
    retrieved: list
    result: dict


def classify_intent_node(state: GraphState):
    intent = classify_intent(state["query"])

    return {
        "intent": intent
    }


def retrieve_and_answer_node(state: GraphState):
    retrieved = retrieve(
        state["query"],
        top_k=3
    )

    result = generate_answer(
        state["query"],
        retrieved
    )

    return {
        "retrieved": retrieved,
        "result": result
    }


def direct_answer_node(state: GraphState):
    result = {
        "answer": (
            "I can help with Zepto customer-support questions. "
            "Please ask about a policy such as delivery, returns, "
            "refunds, cancellation, membership, or payments."
        ),
        "sources": [],
        "confidence": 0.70,
    }

    return {
        "result": result
    }


def route_after_classification(state: GraphState):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


graph_builder = StateGraph(GraphState)

graph_builder.add_node(
    "classify_intent",
    classify_intent_node
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer_node
)

graph_builder.add_node(
    "direct_answer",
    direct_answer_node
)

graph_builder.set_entry_point("classify_intent")

graph_builder.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)

graph_builder.add_edge(
    "retrieve_and_answer",
    END
)

graph_builder.add_edge(
    "direct_answer",
    END
)

graph = graph_builder.compile()


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant is running."
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post(
    "/ask",
    response_model=AnswerResponse
)
def ask(request: AskRequest):

    result = graph.invoke(
        {
            "query": request.query
        }
    )

    return result["result"]