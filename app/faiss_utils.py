import pandas as pd
import os
import pathlib
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document

from app.embeddings import get_openai_embeddings

FAISS_PATH_COCKTAILS = (
    pathlib.Path(__file__).parent.parent / "data" / "faiss" / "vector_cocktails"
)
FAISS_PATH_USERS = (
    pathlib.Path(__file__).parent.parent / "data" / "faiss" / "vector_users"
)
CSV_PATH = (
    pathlib.Path(__file__).parent.parent / "data" / "cocktails" / "final_cocktails.csv"
)


def load_cocktails_csv() -> pd.DataFrame:
    return pd.read_csv(
        pathlib.Path(__file__).parent / "data" / "cocktails" / "final_cocktails.csv"
    )


def get_user_faiss_path(user_id: str) -> str:
    """Returns the path to the user's FAISS index directory."""
    return os.path.join(FAISS_PATH_USERS, user_id)


def create_faiss_index_cocktails(
    source_column: list[str] | str = "name",
    metadata_columns: list[str] | str = ["alcoholic", "category"],
    faiss_path: str = FAISS_PATH_COCKTAILS,
) -> None:
    embeddings_model = get_openai_embeddings()

    document_loader = CSVLoader(
        CSV_PATH, source_column=source_column, metadata_columns=metadata_columns
    )

    docs = document_loader.load_and_split()
    vector_store = FAISS.from_documents(docs, embeddings_model)

    vector_store.save_local(str(faiss_path))


def load_faiss_index(
    embeddings,
    faiss_path: str = FAISS_PATH_COCKTAILS,
    allow_dangerous_deserialization: bool = True,
):
    return FAISS.load_local(
        str(faiss_path),
        embeddings,
        allow_dangerous_deserialization=allow_dangerous_deserialization,
    )


def load_user_faiss_index(user_id: str = "user") -> FAISS:
    """Loads the FAISS index for the given user or creates a new one."""
    user_faiss_path = get_user_faiss_path(user_id)
    embeddings_model = get_openai_embeddings()

    # Create a new FAISS index if none exists
    if not os.path.exists(user_faiss_path):
        os.makedirs(user_faiss_path, exist_ok=True)
        index = FAISS.from_documents(
            [
                Document(
                    page_content="Drink",
                    metadata={"source": f"{user_id}"},
                )
            ],
            embeddings_model,
        )
        index.save_local(user_faiss_path)

    # Load existing FAISS index
    return FAISS.load_local(
        user_faiss_path, embeddings_model, allow_dangerous_deserialization=True
    )


def add_user_info_to_faiss_index(user_id: str, user_info: str) -> None:
    """Adds user information to the FAISS index."""
    user_faiss_path = get_user_faiss_path(user_id)

    index = load_user_faiss_index(user_id)
    index.add_documents(
        [
            Document(
                page_content=user_info,
                metadata={"source": f"{user_id}"},
            )
        ],
    )
    index.save_local(user_faiss_path)


if __name__ == "__main__":
    load_user_faiss_index()
    create_faiss_index_cocktails()
