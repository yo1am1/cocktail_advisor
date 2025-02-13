from langchain_community.vectorstores import FAISS


def find_similar_cocktails(
    user_query: str,
    database: FAISS,
    filter: dict = None,
    limit: int = 4,
    score_threshold: float = 0.76,
):
    return database.similarity_search(
        user_query, score_threshold=score_threshold, filter=filter, k=limit
    )
