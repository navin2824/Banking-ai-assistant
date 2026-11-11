
from src.rag import BankingRAG


def main():

    print("\nBANKING AI ASSISTANT - RAG SMOKE TEST")
    print("=" * 60)

    rag = BankingRAG(
        chunk_size=1200,
        overlap=150
    )

    # Load documents and build index
    rag.load_documents()
    rag.build_index()

    question = "What is a Non-Performing Asset in banking?"

    print("\nQuestion:", question)

    context, results = rag.get_context(
        question,
        top_k=3,
        min_similarity=0.03
    )

    print("\nRetrieved Banking Knowledge")
    print("=" * 60)

    if not results:
        print("No relevant information found.")
        return

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")
        print("Source:", result["source"])
        print("Section:", result["section"])
        print("Chunk ID:", result["chunk_id"])
        print("Similarity:", result["similarity"])
        print("Content:", result["text"][:400])
        print("-" * 60)

    print("\nComplete Context")
    print("=" * 60)
    print(context)

    print("\nRAG SMOKE TEST COMPLETED.")


if __name__ == "__main__":
    main()
