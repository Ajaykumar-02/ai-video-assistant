import os 
from langchain_chroma import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"   # default only; every chat passes its own collection name
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

#Embedding
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" : 'cpu'}
    )


#Vector Delete
def delete_vector_store(collection_name : str = COLLECTION_NAME) -> bool:
    """Delete one collection (all its chunks + embeddings) from the Chroma DB.
    Other chats' collections are not touched."""
    try:
        Chroma(
            collection_name=collection_name,
            persist_directory=CHROMA_DIR
        ).delete_collection()
        print(f"Deleted vector store: {collection_name}")
        return True
    except Exception as e:
        print(f"Could not delete vector store '{collection_name}': {e}")
        return False


#Vector Create
def build_vector_store(transcript : str, collection_name : str = COLLECTION_NAME)->Chroma:
    print(f"Building vector Store: {collection_name}")

    # Start clean: never mix new chunks with data already stored under this name
    delete_vector_store(collection_name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_text(transcript) #text split

    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i})
        for i,chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents= docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR
    )

    return vector_store


# Vector load
def load_vector_store(collection_name : str = COLLECTION_NAME) ->Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function= embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store

# Retriver
def get_retriever(vector_store : Chroma, k :int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )