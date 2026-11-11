
import re
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNK_FILE = PROCESSED_DIR / "chunks.json"


# =====================================================
# BANKING RAG CLASS
# =====================================================

class BankingRAG:

    def __init__(self, chunk_size=1200, overlap=150):

        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive.")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("Invalid overlap value.")

        self.chunk_size = chunk_size
        self.overlap = overlap

        self.chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None

    # =================================================
    # TEXT CLEANING
    # =================================================

    def clean_text(self, text):

        text = text.replace("\ufeff", "")
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    # =================================================
    # HEADING EXTRACTION
    # =================================================

    def extract_heading(self, line):

        line = line.strip()

        numbered = re.match(
            r"^(\d+(?:\.\d+)*\.)\s+(.*)$",
            line
        )

        if numbered:

            number = numbered.group(1)
            body = numbered.group(2).strip()

            boundary = re.search(
                r"\s+(?=[A-Z][a-z]|\b[A-Z]\s+[a-z])",
                body
            )

            if boundary:
                heading_title = body[:boundary.start()].strip()
                remaining_text = body[boundary.end():].strip()
            else:
                heading_title = body
                remaining_text = ""

            if (
                heading_title
                and heading_title == heading_title.upper()
                and re.search(r"[A-Z]", heading_title)
            ):

                heading = f"{number} {heading_title}"

                return heading, remaining_text

        # Standalone uppercase headings

        if (
            len(line) <= 120
            and line.isupper()
            and len(line.split()) <= 12
            and re.search(r"[A-Z]", line)
        ):

            return line, ""

        return None, line

    # =================================================
    # SPLIT LONG TEXT
    # =================================================

    def split_long_text(self, text):

        words = text.split()
        chunks = []

        start = 0

        overlap_words = max(0, self.overlap // 6)
        chunk_words = max(1, self.chunk_size // 6)

        while start < len(words):

            end = min(
                start + chunk_words,
                len(words)
            )

            chunk = " ".join(words[start:end])

            if chunk:
                chunks.append(chunk)

            if end == len(words):
                break

            start = max(
                start + 1,
                end - overlap_words
            )

        return chunks

    # =================================================
    # SECTION-AWARE CHUNKING
    # =================================================

    def split_text(self, text):

        text = self.clean_text(text)

        lines = [
            line.strip()
            for line in text.split("\n")
            if line.strip()
        ]

        sections = []

        current_title = "General Information"
        current_lines = []

        for line in lines:

            heading, remaining_text = self.extract_heading(line)

            if heading:

                if current_lines:

                    sections.append(
                        (current_title, current_lines)
                    )

                current_title = heading
                current_lines = []

                if remaining_text:
                    current_lines.append(remaining_text)

            else:

                current_lines.append(line)

        if current_lines:

            sections.append(
                (current_title, current_lines)
            )

        chunks = []

        for title, section_lines in sections:

            current_chunk = ""

            for line in section_lines:

                if len(line) > self.chunk_size:

                    if current_chunk:

                        chunks.append({
                            "section": title,
                            "text": current_chunk.strip()
                        })

                    current_chunk = ""

                    for part in self.split_long_text(line):

                        chunks.append({
                            "section": title,
                            "text": part
                        })

                    continue

                candidate = (
                    current_chunk + "\n" + line
                    if current_chunk
                    else line
                )

                if len(candidate) <= self.chunk_size:

                    current_chunk = candidate

                else:

                    if current_chunk:

                        chunks.append({
                            "section": title,
                            "text": current_chunk.strip()
                        })

                    current_chunk = line

            if current_chunk:

                chunks.append({
                    "section": title,
                    "text": current_chunk.strip()
                })

        return chunks

    # =================================================
    # LOAD DOCUMENTS
    # =================================================

    def load_documents(self):

        if not RAW_DIR.exists():

            raise FileNotFoundError(
                f"Document folder not found: {RAW_DIR}"
            )

        files = sorted(RAW_DIR.glob("*.txt"))

        if not files:

            raise FileNotFoundError(
                "No TXT documents found in data/raw/"
            )

        self.chunks = []

        for file_path in files:

            print(f"Processing: {file_path.name}")

            text = file_path.read_text(
                encoding="utf-8"
            )

            if not text.strip():
                continue

            document_chunks = self.split_text(text)

            for index, item in enumerate(document_chunks):

                self.chunks.append({
                    "source": file_path.name,
                    "chunk_id": index,
                    "section": item["section"],
                    "text": item["text"]
                })

        if not self.chunks:

            raise ValueError(
                "No usable document chunks found."
            )

        print("\nDocument loading completed.")
        print(f"Documents loaded: {len(files)}")
        print(f"Total chunks created: {len(self.chunks)}")

        return self.chunks

    # =================================================
    # BUILD TF-IDF INDEX
    # =================================================

    def build_index(self):

        if not self.chunks:
            self.load_documents()

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True,
            strip_accents="unicode",
            norm="l2"
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            texts
        )

        PROCESSED_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            CHUNK_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.chunks,
                file,
                indent=4,
                ensure_ascii=False
            )

        print("\nTF-IDF indexing completed.")
        print(
            f"Vocabulary size: {len(self.vectorizer.vocabulary_)}"
        )
        print(
            f"Indexed chunks: {self.tfidf_matrix.shape[0]}"
        )
        print(f"Saved chunks: {CHUNK_FILE}")

    # =================================================
    # RETRIEVE RELEVANT CHUNKS
    # =================================================

    def retrieve(
        self,
        question,
        top_k=3,
        min_similarity=0.01
    ):

        if not question or not question.strip():
            return []

        if top_k < 1:
            return []

        if self.tfidf_matrix is None:
            self.build_index()

        question_vector = self.vectorizer.transform(
            [question.strip()]
        )

        similarities = cosine_similarity(
            question_vector,
            self.tfidf_matrix
        ).flatten()

        results = []

        for index, similarity in enumerate(similarities):

            similarity = float(similarity)

            if similarity < min_similarity:
                continue

            chunk = self.chunks[index]

            results.append({
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "section": chunk["section"],
                "text": chunk["text"],
                "similarity": round(similarity, 4)
            })

        results.sort(
            key=lambda item: item["similarity"],
            reverse=True
        )

        return results[:top_k]

    # =================================================
    # DETECT QUESTIONS REQUIRING LIVE DATA
    # =================================================

    def requires_live_data(self, question):

        text = question.lower()

        time_indicators = [
            "today",
            "current",
            "currently",
            "latest",
            "now",
            "as of today",
            "this week",
            "this month",
            "right now",
            "present rate",
            "up-to-date"
        ]

        live_topics = [
            "repo rate",
            "interest rate",
            "home loan rate",
            "home loan interest",
            "gold price",
            "gold rate",
            "stock market",
            "stock price",
            "market closing",
            "market value",
            "inflation rate",
            "exchange rate",
            "forex rate",
            "sensex",
            "nifty",
            "share price"
        ]

        has_time_indicator = any(
            term in text
            for term in time_indicators
        )

        has_live_topic = any(
            term in text
            for term in live_topics
        )

        return (
            has_time_indicator
            and has_live_topic
        )

    # =================================================
    # ASSESS RETRIEVAL EVIDENCE
    # =================================================

    def assess_retrieval(
        self,
        question,
        top_k=3,
        min_similarity=0.01,
        confidence_threshold=0.05
    ):

        # Reject questions requiring current information.

        if self.requires_live_data(question):

            return {
                "supported": False,
                "confidence": 0.0,
                "reason": (
                    "This question requires current information "
                    "that cannot be verified using static documents."
                ),
                "results": [],
                "requires_live_data": True
            }

        # Retrieve document evidence.

        results = self.retrieve(
            question,
            top_k=top_k,
            min_similarity=min_similarity
        )

        if not results:

            return {
                "supported": False,
                "confidence": 0.0,
                "reason": "No matching evidence found.",
                "results": [],
                "requires_live_data": False
            }

        top_score = results[0]["similarity"]

        # Initial similarity-based confidence gate.
        # This score is not a probability of correctness.

        supported = (
            top_score >= confidence_threshold
        )

        if supported:

            reason = "Evidence passed the initial retrieval threshold."

        else:

            reason = (
                "Retrieved evidence is too weak "
                "to support a reliable answer."
            )

        return {
            "supported": supported,
            "confidence": top_score,
            "reason": reason,
            "results": results,
            "requires_live_data": False
        }

    # =================================================
    # GENERATE CONTEXT FOR NUGEN
    # =================================================

    def get_context(
        self,
        question,
        top_k=3,
        min_similarity=0.01,
        confidence_threshold=0.05
    ):

        assessment = self.assess_retrieval(
            question,
            top_k=top_k,
            min_similarity=min_similarity,
            confidence_threshold=confidence_threshold
        )

        results = assessment["results"]

        if not assessment["supported"]:

            return (
                assessment["reason"],
                []
            )

        context_parts = []

        for index, result in enumerate(results, start=1):

            context_parts.append(
                f"""
SOURCE {index}
DOCUMENT: {result['source']}
SECTION: {result['section']}
CHUNK ID: {result['chunk_id']}
SIMILARITY: {result['similarity']}

CONTENT:
{result['text']}
"""
            )

        return "\n".join(context_parts), results

    # =================================================
    # DISPLAY RETRIEVAL RESULTS
    # =================================================

    def display_results(self, results):

        if not results:

            print("\nNo relevant documents found.")
            return

        print("\nRETRIEVED DOCUMENTS")
        print("=" * 70)

        for index, result in enumerate(results, start=1):

            print(f"\nResult {index}")
            print("Source:", result["source"])
            print("Section:", result["section"])
            print("Chunk ID:", result["chunk_id"])
            print("Similarity:", result["similarity"])
            print("Content:", result["text"][:500])
            print("-" * 70)


# =====================================================
# MAIN TESTING
# =====================================================

if __name__ == "__main__":

    rag = BankingRAG(
        chunk_size=1200,
        overlap=150
    )

    rag.load_documents()
    rag.build_index()

    test_questions = [
        "What is a Non-Performing Asset in Indian banking?",
        "What is the role of the Reserve Bank of India?",
        "Explain credit risk in banking.",
        "What is liquidity risk?",
        "How does EMI repayment work?",
        "What are the responsibilities of RBI?",
        "Explain how loan principal and interest are repaid.",
        "What is the current RBI repo rate today?",
        "What is today's gold price in India?"
    ]

    for question in test_questions:

        print("\n\nQUESTION")
        print("=" * 70)
        print(question)

        assessment = rag.assess_retrieval(
            question,
            top_k=3
        )

        print("\nRETRIEVAL ASSESSMENT")
        print("Supported:", assessment["supported"])
        print("Confidence:", assessment["confidence"])
        print("Reason:", assessment["reason"])
        print(
            "Requires live data:",
            assessment["requires_live_data"]
        )

        rag.display_results(
            assessment["results"]
        )

    print("\n\nRAG TESTING COMPLETED.")
