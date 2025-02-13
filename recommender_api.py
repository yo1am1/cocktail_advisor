from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import Literal, Dict
from icecream import ic
from langchain_core.tools import tool
from pathlib import Path

from app.faiss_utils import (
    load_faiss_index,
    add_user_info_to_faiss_index,
    load_user_faiss_index,
)
from app.embeddings import (
    get_openai_embeddings,
    get_openai_model,
)

from langchain_core.output_parsers import StrOutputParser

from app.prompts import recommender

embeddings_model = get_openai_embeddings()
llm = get_openai_model()

faiss_index = load_faiss_index(
    embeddings_model,
    faiss_path=str(Path(__file__).parent / "data" / "faiss" / "vector_cocktails"),
)

app = FastAPI(
    title="Cocktail Advisor",
    description="A simple API that provides cocktail recommendations.",
    version="0.1",
    openapi_tags=[
        {
            "name": "chat",
            "description": "Chatbot API",
        },
        {
            "name": "vector_store",
            "description": "Vector Store API",
        },
    ],
)


class CocktailQuery(BaseModel):
    user_id: str = Field(title="User name", description="Unique username for the user.")
    user_query: str = Field(
        min_length=1,
        max_length=1000,
        title="User Query",
        description="User's query to find the cocktail they want.",
    )


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        title="Question",
        description="The question to ask the chatbot.",
    )
    context: str | None = Field(
        min_length=1,
        max_length=100000,
        title="Context",
        description="The context to provide the chatbot.",
    )
    history: str | None = Field(
        min_length=0,
        max_length=100000,
        title="History",
        description="The history of the conversations.",
    )
    model: Literal["gpt-4o", "gpt-4o-mini"]


class MemoryRequest(BaseModel):
    user_id: str = Field(
        ..., title="User ID", description="Unique identifier for the user."
    )
    memory_text: str = Field(
        ...,
        title="Memory",
        description="User's memory (favorite ingredients/cocktails) to store.",
    )


@tool
def evaluate_user_memory_action(
    required_action: Literal["store", "retrieve"]
) -> Dict[str, str]:
    """
    This returns whether the user's query should be stored as a new memory or used for retrieval based on the user's input.
    Returns a JSON object with the following structure:
    {
        "action": action
    }

    Examples:
        User input: "I love cocktails with mint and lime"
        Output: {"action": "store"}
        ---
        User input: "What are some refreshing cocktails?"
        {"action": "retrieve"}
        ---
        User input: "I like cocktails with mint and lime."
        {"action": "store"}

    Args:
        required_action: The required action to be taken. One of ['store', 'retrieve']
    """


@tool
def get_preferences_from_query(preferred_ingredients: list[str]) -> list[str]:
    """
    This function returns the user's ingredient preferences based on the user's input.

    Examples:
        User input: "give me 3 cocktails with 'Rum', 'Lemon-lime soda'"
        Output: ['Rum', 'Lemon-lime soda']
        ---
        User input: "I like cocktails with mint and lime."
        Output: ['Mint', 'Lime']
        ---
        User input: "Suggest the cocktail with Milk"
        Output: ['Milk']

    Args:
        preferred_ingredients: The user's preferred ingredients.
    """


@tool
def get_optimal_metadata_field_info(
    category: str, alcoholic: str, k_results: int
) -> Dict[str, str]:
    """
    This function returns the optimal metadata field info and k results for the recommender based on user input and User memory.
    Metadata fields are used to filter results in vector store and k results is the number of results to return.

    Examples:
        User input: What are the 5 cocktails containing lemon?
        Output: {
            "category": "Cocktail",
            "alcoholic": "Alcoholic",
            "k_results": 5
        }
        User input: Give me the cocktail with rum and lemon
        Output: {
            "category": "Cocktail",
            "alcoholic": "Alcoholic",
            "k_results": 1
        }

    Args:
        category: One of ['Cocktail', 'Shot', 'Ordinary Drink', 'Other / Unknown', 'Coffee / Tea', 'Beer', 'Punch / Party Drink', 'Shake', 'Soft Drink', 'Homemade Liqueur', 'Cocoa']
        alcoholic: One of ['Alcoholic', 'Non-Alcoholic', 'Optional alcohol']
        k_results: The number of results.
    """


@app.post("/store_user_memory", tags=["vector_store"])
async def store_user_memory(request: MemoryRequest) -> JSONResponse:
    """
    Endpoint to store user's favorite ingredients or cocktail preferences in a user-specific FAISS index.
    
    Args:
        request: MemoryRequest object containing user_id and memory_text.
        
    Returns:
        JSONResponse: object with a message indicating if the memory was stored successfully.
    """
    try:
        add_user_info_to_faiss_index(request.user_id, request.memory_text)
        return JSONResponse(
            content={
                "message": "User memory stored successfully",
                "user_id": request.user_id,
            },
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def redirect_to_streamlit():
    """
    Redirects to the Streamlit app.

    Returns:
        RedirectResponse: Redirects to the Streamlit app.
    """
    return RedirectResponse(url="http://localhost:8501", status_code=302)


@app.post("/chat", tags=["chat"])
async def chat(request: QueryRequest) -> JSONResponse:
    """
    Endpoint to chat with the chatbot.

    Args:
        request (QueryRequest): The request object containing the question, context, history, and model.

    Returns:
        JSONResponse: The response from the chatbot.
    """
    prompt = recommender.copy()
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke(
        {
            "question": request.question,
            "context": request.context,
            "history": request.history,
        }
    )
    return JSONResponse(content={"response": response}, status_code=200)


@app.post("/retrieve_recommendations", tags=["vector_store"])
async def recommend_cocktails(query: CocktailQuery) -> JSONResponse:
    """
    Endpoint to retrieve cocktail recommendations based on user query and user memory.

    Args:
        query (CocktailQuery): The user query and user ID.

    Raises:
        HTTPException: If no context is found.

    Returns:
        JSONResponse: The response containing the cocktail recommendations.
    """
    eval_ingr = llm.bind_tools(
        [get_preferences_from_query], tool_choice="get_preferences_from_query"
    )
    eval_mem = llm.bind_tools(
        [evaluate_user_memory_action], tool_choice="evaluate_user_memory_action"
    )
    eval_result_mem = eval_mem.invoke(query.user_query)
    eval_result_ingr = eval_ingr.invoke(query.user_query)

    for tool_call in eval_result_ingr.tool_calls:
        if tool_call["name"] == "get_preferences_from_query":
            tool_args = tool_call["args"]
            pref_ingr = tool_args.get("preferred_ingredients", None)

    for tool_call in eval_result_mem.tool_calls:
        if tool_call["name"] == "evaluate_user_memory_action":
            tool_args = tool_call["args"]
            action = tool_args.get("required_action", "retrieve")

    user_memory_context = "User memory:"
    if action == "store":
        user_pref = query.user_query
        try:
            add_user_info_to_faiss_index(query.user_id, user_pref)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail="Error storing new user memory."
            )
    else:
        user_index = load_user_faiss_index(query.user_id)
        user_pref = user_index.similarity_search(query.user_query, score_threshold=0.7)

    user_memory_context += (
        "; ".join([user.page_content for user in user_pref])
        if isinstance(user_pref, list)
        else user_pref
    )

    selfq_retriever = llm.bind_tools(
        [get_optimal_metadata_field_info], tool_choice="get_optimal_metadata_field_info"
    ).invoke(query.user_query + user_memory_context)

    for tool_call in selfq_retriever.tool_calls:
        if tool_call["name"] == "get_optimal_metadata_field_info":
            tool_args = tool_call["args"]
            cat_f = {"category": {"$eq": tool_args.get("category", None)}}
            alcoh_f = {"alcoholic": {"$eq": tool_args.get("alcoholic", None)}}
            ingr_f = {"ingredients": {"$in": pref_ingr}}
            k_results = tool_args.get("k_results", 4)

    combined_filter = {"$or": [cat_f, alcoh_f, ingr_f]}

    context = faiss_index.search(
        query.user_query, filter=combined_filter, k=k_results, search_type="mmr"
    )

    if not context:
        raise HTTPException(
            status_code=404,
            detail="No context found.",
        )

    formatted_results = []
    for doc in context:
        formatted_results.append(
            {
                "name": doc.metadata.get("source", "Unknown"),
                "ingredients": doc.page_content.split("ingredients: ")[1].split("\n")[0]
                if "ingredients: " in doc.page_content
                else "N/A",
                "instructions": doc.page_content.split("instructions: ")[1].split("\n")[
                    0
                ]
                if "instructions: " in doc.page_content
                else "N/A",
                "category": doc.metadata.get("category", "Unknown"),
                "alcoholic": doc.metadata.get("alcoholic", "Unknown"),
            }
        )
    formatted_results.append(user_memory_context)

    return JSONResponse(content={"response": formatted_results}, status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("recommender_api:app", host="127.0.0.1", port=8000, reload=True)
