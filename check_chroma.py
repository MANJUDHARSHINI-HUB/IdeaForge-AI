import chromadb

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    "ideaforge_knowledge_v2"
)

data = collection.get(
    where={"type": "project_idea"},
    include=["documents"]
)

print("\n====================================")
print("PROJECT 1 STORED IN CHROMADB")
print("====================================\n")

print(data["documents"][0])

print("\n====================================")
input("Press Enter to close...")