import subprocess
from bs4 import BeautifulSoup

# =========================================
# LOAD KNOWLEDGE BASE
# =========================================

with open("docs/knowledge_base.html", "r", encoding="utf-8") as f:

    soup = BeautifulSoup(f, "lxml")

    knowledge = soup.get_text(separator=" ", strip=True)

print("\n===================================")
print("AUTOMOTIVE RAG ASSISTANT READY")
print("===================================\n")

# =========================================
# CHAT LOOP
# =========================================

while True:

    query = input("Ask: ")

    if query.lower() == "exit":
        break

    # =========================================
    # BUILD PROMPT
    # =========================================

    prompt = f"""
You are an intelligent automotive AI assistant.

You MUST answer ONLY using the provided knowledge base.

If the answer does not exist in the knowledge base, reply:
"Information not available in local knowledge base."

Knowledge Base:
{knowledge}

User Question:
{query}

Instructions:
- answer clearly
- answer briefly
- use technical explanations when needed
- use driver profile data if relevant
"""

    # =========================================
    # RUN QWEN
    # =========================================

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

    print("\n===================================")
    print("QWEN RESPONSE")
    print("===================================\n")

    print(result.stdout)
