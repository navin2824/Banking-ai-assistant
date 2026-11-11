
from src.rag import BankingRAG


# =====================================================
# BANKING RAG EVALUATION DATASET
# =====================================================

TEST_CASES = [
    {
        "question": "What is a Non-Performing Asset in Indian banking?",
        "expected_sources": [
            "01_banking_fundamentals.txt",
            "02_rbi_regulatory_and_asset_quality.txt"
        ],
        "expected_sections": [
            "5. ASSET QUALITY AND NON-PERFORMING ASSETS",
            "5. NON-PERFORMING ASSETS"
        ],
        "expected_keywords": [
            "non-performing asset",
            "prudential",
            "classification",
            "asset quality"
        ],
        "answerable": True
    },
    {
        "question": "What is the role of the Reserve Bank of India?",
        "expected_sources": [
            "02_rbi_regulatory_and_asset_quality.txt"
        ],
        "expected_sections": [
            "1. ROLE OF THE RESERVE BANK OF INDIA"
        ],
        "expected_keywords": [
            "monetary policy",
            "currency",
            "supervision",
            "reserve bank"
        ],
        "answerable": True
    },
    {
        "question": "Explain credit risk in banking.",
        "expected_sources": [
            "03_financial_risk_and_reliability.txt"
        ],
        "expected_sections": [
            "1. CREDIT RISK"
        ],
        "expected_keywords": [
            "borrower",
            "counterparty",
            "contractual obligations"
        ],
        "answerable": True
    },
    {
        "question": "What is liquidity risk?",
        "expected_sources": [
            "03_financial_risk_and_reliability.txt"
        ],
        "expected_sections": [
            "2. LIQUIDITY RISK"
        ],
        "expected_keywords": [
            "cash-flow obligations",
            "liquid-asset buffers",
            "funding",
            "liquidity"
        ],
        "answerable": True
    },
    {
        "question": "How does EMI repayment work?",
        "expected_sources": [
            "01_banking_fundamentals.txt"
        ],
        "expected_sections": [
            "3. LENDING TERMINOLOGY",
            "4. HOW LOAN REPAYMENT WORKS"
        ],
        "expected_keywords": [
            "principal",
            "interest",
            "instalment",
            "emi"
        ],
        "answerable": True
    },
    {
        "question": "Does a high credit score guarantee loan approval?",
        "expected_sources": [
            "01_banking_fundamentals.txt"
        ],
        "expected_sections": [
            "3. LENDING TERMINOLOGY"
        ],
        "expected_keywords": [
            "credit score",
            "income",
            "underwriting",
            "loan approval"
        ],
        "answerable": True
    }
]


UNANSWERABLE_CASES = [
    "What is the current RBI repo rate today?",
    "What is the current SBI home loan interest rate?",
    "What is today's gold price in India?",
    "What is the latest stock market closing value?",
    "What is the current inflation rate in India?"
]


# =====================================================
# EVALUATION CONFIGURATION
# =====================================================

TOP_K = 3
MIN_SIMILARITY = 0.03


# =====================================================
# RETRIEVAL RELEVANCE CHECK
# =====================================================

def is_relevant(result, test_case):

    source_matches = (
        result["source"] in test_case["expected_sources"]
    )

    expected_sections = test_case.get("expected_sections", [])

    section_matches = (
        not expected_sections
        or result["section"] in expected_sections
    )

    text = result["text"].lower()

    keyword_matches = any(
        keyword.lower() in text
        for keyword in test_case["expected_keywords"]
    )

    return (
        source_matches
        and section_matches
        and keyword_matches
    )


# =====================================================
# EVALUATE ANSWERABLE QUESTIONS
# =====================================================

def evaluate_answerable(rag):

    print("\n" + "=" * 75)
    print("ANSWERABLE QUESTION EVALUATION")
    print("=" * 75)

    precision_scores = []
    hit_scores = []
    reciprocal_ranks = []

    for index, test_case in enumerate(TEST_CASES, start=1):

        question = test_case["question"]

        results = rag.retrieve(
            question,
            top_k=TOP_K,
            min_similarity=MIN_SIMILARITY
        )

        relevant_count = sum(
            is_relevant(result, test_case)
            for result in results
        )

        # Precision@K uses K as the denominator,
        # even if fewer than K chunks are returned.
        precision = relevant_count / TOP_K

        relevant_ranks = [
            rank
            for rank, result in enumerate(results, start=1)
            if is_relevant(result, test_case)
        ]

        hit = 1 if relevant_ranks else 0

        reciprocal_rank = (
            1 / relevant_ranks[0]
            if relevant_ranks else 0
        )

        precision_scores.append(precision)
        hit_scores.append(hit)
        reciprocal_ranks.append(reciprocal_rank)

        print(f"\nTest {index}: {question}")
        print("Expected sources:", test_case["expected_sources"])
        print("Retrieved chunks:", len(results))

        for rank, result in enumerate(results, start=1):

            relevant = is_relevant(result, test_case)

            print(
                f"  Rank {rank}: "
                f"{result['source']} | "
                f"Section: {result['section']} | "
                f"Similarity: {result['similarity']:.4f} | "
                f"Relevant: {relevant}"
            )

        print(f"Precision@{TOP_K}: {precision:.3f}")
        print(f"Hit@{TOP_K}: {hit}")
        print(f"Reciprocal Rank: {reciprocal_rank:.3f}")

    average_precision = sum(precision_scores) / len(precision_scores)
    hit_rate = sum(hit_scores) / len(hit_scores)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("\nANSWERABLE EVALUATION SUMMARY")
    print("-" * 50)
    print(f"Precision@{TOP_K}: {average_precision:.3f}")
    print(f"Hit Rate@{TOP_K}: {hit_rate:.3f}")
    print(f"MRR: {mrr:.3f}")

    return {
        "precision_at_k": average_precision,
        "hit_rate_at_k": hit_rate,
        "mrr": mrr
    }


# =====================================================
# EVALUATE UNANSWERABLE QUESTIONS
# =====================================================

def evaluate_unanswerable(rag):

    print("\n" + "=" * 75)
    print("UNANSWERABLE QUESTION EVALUATION")
    print("=" * 75)

    rejected_count = 0

    for index, question in enumerate(UNANSWERABLE_CASES, start=1):

        assessment = rag.assess_retrieval(
            question,
            top_k=TOP_K,
            min_similarity=MIN_SIMILARITY,
            confidence_threshold=0.05
        )

        rejected = not assessment["supported"]

        if rejected:
            rejected_count += 1

        print(f"\nTest {index}: {question}")
        print("Supported:", assessment["supported"])
        print("Requires live data:", assessment["requires_live_data"])
        print("Reason:", assessment["reason"])

        if assessment["results"]:
            print("Retrieved chunks:")

            for result in assessment["results"]:
                print(
                    f"  {result['source']} | "
                    f"{result['section']} | "
                    f"Similarity: {result['similarity']}"
                )
        else:
            print("Retrieved chunks: 0")

    rejection_rate = (
        rejected_count / len(UNANSWERABLE_CASES)
    )

    print("\nUNANSWERABLE EVALUATION SUMMARY")
    print("-" * 50)
    print(f"Questions tested: {len(UNANSWERABLE_CASES)}")
    print(f"Rejected questions: {rejected_count}")
    print(f"Rejection rate: {rejection_rate:.3f}")

    return {
        "rejection_rate": rejection_rate
    }


# =====================================================
# MAIN EVALUATION
# =====================================================

if __name__ == "__main__":

    print("\nBANKING AI ASSISTANT")
    print("AUTOMATED RAG EVALUATION")

    rag = BankingRAG(
        chunk_size=1200,
        overlap=150
    )

    rag.load_documents()
    rag.build_index()

    answerable_results = evaluate_answerable(rag)

    unanswerable_results = evaluate_unanswerable(rag)

    print("\n" + "=" * 75)
    print("FINAL EVALUATION REPORT")
    print("=" * 75)

    print(
        f"Precision@{TOP_K}: "
        f"{answerable_results['precision_at_k']:.3f}"
    )

    print(
        f"Hit Rate@{TOP_K}: "
        f"{answerable_results['hit_rate_at_k']:.3f}"
    )

    print(
        f"Mean Reciprocal Rank: "
        f"{answerable_results['mrr']:.3f}"
    )

    print(
        f"Retrieval rejection rate: "
        f"{unanswerable_results['rejection_rate']:.3f}"
    )

    print("\nEVALUATION COMPLETED.")
