from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from decouple import config


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

combine_template = "Write a summary of the following text:\n\n{summaries}"
combine_prompt_template = PromptTemplate.from_template(
    template=combine_template)

question_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
question_prompt_template = PromptTemplate.from_template(
    template=question_template)

# create chat model
llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), temperature=0)

# create retriever chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    # mmr > for diversity in documents
    # Set fetch_k value to get the fetch_k most similar search. This is basically semantic search
    retriever=vector_db.as_retriever(
        search_kwargs={'fetch_k': 4, 'k': 3}, search_type='mmr'),
    return_source_documents=True,
    chain_type="map_reduce",
    chain_type_kwargs={"question_prompt": question_prompt_template,
                       "combine_prompt": combine_prompt_template}
)

# question
question = "What areas is Python mostly used"

# call QA chain
response = qa_chain({"query": question})

print(response)

print("============================================")
print("====================Result==================")
print("============================================")
print(response["result"])


print("============================================")
print("===============Source Documents============")
print("============================================")

print(response["source_documents"][0])
