from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Removed unused import as it is not resolved
#print("=====Chroma Vector Store=====")
#persist_directory = "../data/vector_db/chroma_db"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

os.environ["LANGSMITH_TRACING"] = "true"

file_path = "../data/pdf/ChatGPTPrompts.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))
#print(len(docs))
print("=====content=====")
print(f"{docs[0].page_content[:200]}\n")
print("=====metadata=====")
print(docs[0].metadata)

embeddings = OllamaEmbeddings(
    #model="qwen2.5:7b-instruct",
    model="nomic-embed-text:latest",
    base_url="http://192.168.0.195:11434"
#    model_kwargs={"num_ctx": 2048}
)

#vector_1 = embeddings.embed_query(all_splits[0].page_content)
#vector_2 = embeddings.embed_query(all_splits[1].page_content)


#assert len(vector_1) == len(vector_2)
#print(f"Generated vectors of length {len(vector_1)}\n")
#print(vector_1[:10])

print("=====InMemoryVectorStore=====")
vector_store = InMemoryVectorStore(embeddings)
#vector_store = Chroma(
#    collection_name="pdf_documents",
#    embedding_function=embeddings,
#    client_settings=chromadb.config.Settings(
#        chroma_api_impl="rest",
#        chroma_server_host="localhost",
#        chroma_server_http_port="8000"
#    )
#)

#print("Adding documents to Chroma vector store...")
ids = vector_store.add_documents(documents=all_splits)
#vector_store.persist()
print(f"Saved {len(ids)} documents to Chroma\n")



#batch_size = 50
#for i in range(0, len(all_splits), batch_size):
#    batch = all_splits[i:i + batch_size]
#    vector_store.add_documents(batch)
#    print(f"Processed {i + len(batch)} / {len(all_splits)}")





llm = ChatOllama(
    model="qwen2.5:7b-instruct", # Or your preferred model
    base_url="http://192.168.0.195:11434", # Replace with your remote IP
    temperature=0
)

print("querying vector store...")
#results = vector_store.similarity_search(
#    "How can I use ChatGPT prompts effectively?", k=3
#)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "\n\n"
    "{context}"
)
input = "How can I use ChatGPT prompts effectively?"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Ask a question!
result = rag_chain.invoke(
    #{"input": "What does the document say about [Your Topic]?"}
     "What does the document say about prompt?"
)

print("=====similarity search results=====")
print(result)
