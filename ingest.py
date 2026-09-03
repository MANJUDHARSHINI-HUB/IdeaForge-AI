import os
import chromadb
import ollama

# ---------------------------------------
# ChromaDB
# ---------------------------------------

client = chromadb.PersistentClient(path="chroma_db")

# Delete old collection if it exists
try:
    client.delete_collection("ideaforge_knowledge_v2")
    print("Old knowledge collection deleted.")
except:
    print("No old collection found.")

# Create a fresh collection
collection = client.create_collection(
    name="ideaforge_knowledge_v2"
)

# ---------------------------------------
# Knowledge Folder
# ---------------------------------------

knowledge_folder = "knowledge"

# ---------------------------------------
# Read Knowledge Files
# ---------------------------------------

for filename in os.listdir(knowledge_folder):

    filepath = os.path.join(
        knowledge_folder,
        filename
    )

    if not filename.endswith(".txt"):
        continue

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    # -----------------------------------
    # Special handling for project ideas
    # -----------------------------------

    if filename == "project_ideas.txt":

        projects = text.split("PROJECT IDEA")

        project_number = 0

        for project in projects:

            if not project.strip():
                continue

            project_number += 1

            chunk = "PROJECT IDEA" + project

            embedding = ollama.embeddings(
                model="nomic-embed-text",
                prompt=chunk
            )["embedding"]

            collection.add(
                ids=[
                    f"project_idea_{project_number}"
                ],
                embeddings=[
                    embedding
                ],
                documents=[
                    chunk
                ],
                metadatas=[
                    {
                        "source": filename,
                        "chunk": project_number - 1,
                        "type": "project_idea"
                    }
                ]
            )

            print(
                f"Added: PROJECT IDEA {project_number}"
            )

    # -----------------------------------
    # Other knowledge files
    # -----------------------------------

    else:

        chunk_size = 1000

        chunks = []

        for i in range(
            0,
            len(text),
            chunk_size
        ):

            chunk = text[i:i + chunk_size]

            if chunk.strip():
                chunks.append(chunk)

        for index, chunk in enumerate(chunks):

            embedding = ollama.embeddings(
                model="nomic-embed-text",
                prompt=chunk
            )["embedding"]

            collection.add(
                ids=[
                    f"{filename}_{index}"
                ],
                embeddings=[
                    embedding
                ],
                documents=[
                    chunk
                ],
                metadatas=[
                    {
                        "source": filename,
                        "chunk": index,
                        "type": "guideline"
                    }
                ]
            )

            print(
                f"Added: {filename} | Chunk: {index}"
            )

# ---------------------------------------
# Finished
# ---------------------------------------

print()
print("===================================")
print("Knowledge base created successfully!")
print("===================================")