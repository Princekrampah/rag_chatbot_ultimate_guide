from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma


TEXT = ["Python is a versatile and widely used programming language known for its clean and readable syntax, which relies on indentation for code structure",
        "It is a general-purpose language suitable for web development, data analysis, AI, machine learning, and automation. Python offers an extensive standard library with modules covering a broad range of tasks, making it efficient for developers.",
        "It is cross-platform, running on Windows, macOS, Linux, and more, allowing for broad application compatibility."
        "Python has a large and active community that develops libraries, provides documentation, and offers support to newcomers.",
        "It has particularly gained popularity in data science and machine learning due to its ease of use and the availability of powerful libraries and frameworks."]

meta_data = [{"source": "document 1", "page": 1},
             {"source": "document 2", "page": 2},
             {"source": "document 3", "page": 3},
             {"source": "document 4", "page": 4}]

embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma.from_texts(
    texts=TEXT,
    embedding=embedding_function,
    metadatas=meta_data
)

# Run using MMR
response = vector_db.max_marginal_relevance_search(
    query="Tell me about a programming language used for data science", k=2, fetch_k=3)

print(response)