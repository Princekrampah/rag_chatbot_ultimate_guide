from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma


PDF_PATH = "../documents/Rich-Dad-Poor-Dad.pdf"

# create loader
loader = PyPDFLoader(PDF_PATH)

pages = loader.load_and_split()

embedding_func = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# create vector store
vectordb = Chroma.from_documents(
    documents=pages,
    embedding=embedding_func,
    persist_directory=f"vector_db",
    collection_name="rich_dad_poor_dad")

# make vector store persistant
vectordb.persist()
