import os
import subprocess
from bs4 import BeautifulSoup

# =========================
# LOAD + PARSE DOCUMENTS
# =========================

chunks = []

print("\nLoading documentation...\n")

for root, dirs, files in os.walk("docs"):

    for file in files:

        if file.endswith(".html"):

            path = os.path.join(root, file)

            try:
                with open(path, "r", encoding="utf-8") as f:

                    soup = BeautifulSoup(f, "lxml")

                    # REMOVE BAD TAGS
                    for tag in soup(["script", "style", "nav"]):
                        tag.decompose()

                    text = soup.get_text(separator=" ", strip=True)

                    # CLEAN TEXT
                    text = text.replace("\n", " ")
                    text = text.replace("\t", " ")

                    while "  " in text:
                        text = text.replace("  ", " ")

                    # BETTER CHUNKING
                    split_chunks = text.split(". ")

                    current_chunk = ""

                    for sentence in split_chunks:

                        current_chunk += sentence + ". "

                        # CREATE MEDIUM-SIZED CHUNKS
                        if len(current_chunk) > 500:

                            chunks.append({
                                "file": path,
                                "chunk": current_chunk
                            })

                            current_chunk = ""

            except Exception as e:
                print(f"Error reading {path}: {e}")

print(f"\nLoaded {len(chunks)} chunks.\n")


# =========================
# SEARCH FUNCTION
# =========================

def retrieve_best_chunk(query):

    query_words = query.lower().split()

    best_score = 0
    best_chunk = None

    for item in chunks:

        chunk_lower = item["chunk"].lower()

        score = 0

        for word in query_words:

            if len(word) > 2 and word in chunk_lower:
                score += 1

        # BOOST IMPORTANT TECHNICAL WORDS
        important_words = [
            "stream",
            "video",
            "image",
            "display",
            "upload",
            "projection",
            "api",
            "socket",
            "command",
            "laser",
            "sequence"
        ]

        for word in important_words:

            if word in query.lower() and word in chunk_lower:
                score += 2

        if score > best_score:

            best_score = score
            best_chunk = item

    return best_chunk, best_score


# =========================
# CHAT LOOP
# =========================

while True:

    query = input("\nAsk: ")

    if query.lower() == "exit":
        break

    retrieved, score = retrieve_best_chunk(query)

    if not retrieved or score < 2:

        print("\nNo relevant documentation found.\n")
        continue

    print("\n==============================")
    print("RETRIEVED DOCUMENT")
    print("==============================")

    print("\nFILE:")
    print(retrieved["file"])

    print("\nMATCH SCORE:", score)

    print("\n==============================")
    print("RETRIEVED CONTEXT")
    print("==============================\n")

    print(retrieved["chunk"][:1500])

    # =========================
    # BUILD PROMPT
    # =========================

    prompt = f"""
You are an embedded AI technical assistant.

Use ONLY the provided documentation context.

If the answer is not present in the context, say:
"Information not found in documentation."

Documentation Context:
{retrieved["chunk"]}

User Question:
{query}

Provide:
- short technical explanation
- function names if available
- concise answer
"""

    # =========================
    # RUN QWEN
    # =========================

    result = subprocess.run(
        [
            "/data/data/com.termux/files/home/llama.cpp/build/bin/llama-simple",
            "-m",
            "/data/data/com.termux/files/home/llama.cpp/models/qwen2-0_5b-instruct-q4_k_m.gguf",
            "-p",
            prompt
        ],
        capture_output=True,
        text=True
    )

    print("\n==============================")
    print("QWEN ANSWER")
    print("==============================\n")

    print(result.stdout)
