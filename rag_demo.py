import os
from bs4 import BeautifulSoup

chunks = []

# LOAD ALL HTML FILES
for root, dirs, files in os.walk("docs"):

    for file in files:

        if file.endswith(".html"):

            path = os.path.join(root, file)

            try:
                with open(path, "r", encoding="utf-8") as f:

                    soup = BeautifulSoup(f, "lxml")

                    text = soup.get_text(separator=" ", strip=True)

                    # SIMPLE CHUNKING
                    split_chunks = text.split(". ")

                    for chunk in split_chunks:

                        if len(chunk) > 100:

                            chunks.append({
                                "file": path,
                                "chunk": chunk
                            })

            except Exception as e:
                print(f"Error: {path} -> {e}")

print(f"\nTotal chunks created: {len(chunks)}")

# SIMPLE SEARCH LOOP
while True:

    query = input("\nAsk something: ").lower()

    if query == "exit":
        break

    found = False

    for item in chunks:

        if query in item["chunk"].lower():

            print("\n======================")
            print("FILE:", item["file"])
            print("======================\n")

            print(item["chunk"][:1000])

            found = True

            break

    if not found:
        print("\nNo relevant chunk found.")
