import os
from bs4 import BeautifulSoup

all_docs = []

for root, dirs, files in os.walk("docs"):
    for file in files:
        if file.endswith(".html"):

            path = os.path.join(root, file)

            try:
                with open(path, "r", encoding="utf-8") as f:

                    soup = BeautifulSoup(f, "lxml")

                    text = soup.get_text(separator=" ", strip=True)

                    all_docs.append({
                        "file": path,
                        "text": text
                    })

            except Exception as e:
                print(f"Error reading {path}: {e}")

print(f"\nTotal HTML files loaded: {len(all_docs)}")

print("\n============================")
print("FIRST DOCUMENT:")
print("============================\n")

print("FILE:", all_docs[0]["file"])

print("\nTEXT SAMPLE:\n")

print(all_docs[0]["text"][:2000])