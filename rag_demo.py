import os
import subprocess
from bs4 import BeautifulSoup

chunks = []

# LOAD + CHUNK HTML DOCS
for root, dirs, files in os.walk("docs"):

    for file in files:

        if file.endswith(".html"):

            path = os.path.join(root, file)

            try:
                with open(path, "r", encoding="utf-8") as f:

                    soup = BeautifulSoup(f, "lxml")

                    text = soup.get_text(separator=" ", strip=True)

                    split_chunks = text.split(". ")

                    for chunk in split_chunks:

                        if len(chunk) > 100:

                            chunks.append({
                                "file": path,
                                "chunk": chunk
                            })

            except Exception as e:
                print(f"Error reading {path}: {e}")

print(f"\nLoaded {len(chunks)} chunks.\n")

# CHAT LOOP
while True:

    query = input("Ask: ")

    if query.lower() == "exit":
        break

    retrieved = None

    # SIMPLE KEYWORD SEARCH
    for item in chunks:

        if query.lower() in item["chunk"].lower():

            retrieved = item
            break

    if not retrieved:

        print("\nNo relevant documentation found.\n")
        continue

    print("\n[Retrieved Context]")
    print(retrieved["chunk"][:500])

    # BUILD PROMPT
    prompt = f"""
Use the following documentation context to answer the question.

Context:
{retrieved["chunk"]}

Question:
{query}

Answer briefly and accurately.
"""

    # RUN QWEN
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

    print("\n[Qwen Answer]\n")

    print(result.stdout)
