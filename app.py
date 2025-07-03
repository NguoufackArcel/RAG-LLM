import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

# Load environment variables from .env file
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY2")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key,
    model_name="text-embedding-3-small"
)   # this is the embedding function for ChromaDB

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient( path="chroma_persistent_db" )

collection_name = "document_collection"
# Get or create a collection in ChromaDB
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=openai_ef
)

# Initialize OpenAI client
client = OpenAI(api_key=openai_key) 

#resp = client.chat.completions.create(
#    model="gpt-3.5-turbo",
#    messages=[
#              {"role": "user", "content": "What is human life experience?"} ]
#)

# Function to add a document to the collection
def add_document_to_collection_from_directory(directory_path):
    print("===Loading documents from directory ===")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path , 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append({
                    "id": filename,
                    "text": content
                })
    return documents

# function to split documents into chunks
def split_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Load documents from the specified directory
directory_path = "./news_articles"
documents = add_document_to_collection_from_directory(directory_path)

print(F"===Loaded {len(documents)} documents from directory=== ")

# Split documents into chunks and add them to the collection
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    for i, chunk in enumerate(chunks):
        chunked_documents.append({
            "id": f"{doc['id']}_chunk_{i}",
            "text": chunk
        })

print(F"===Splitting documents into {len(chunked_documents)} chunks===")

# function to generate embeddings for the documents using the OpenAI embedding function
def generate_embeddings(text):
    print("===Generating embeddings ..===")
    response = client.embeddings.create( input=text, model="text-embedding-3-small" )
    embedding = response.data[0].embedding
    return embedding

# Generate embeddings for the chunked documents
for doc in chunked_documents:
    print(F"===Generating embedding for {doc['id']} ===")
    # Generate embedding for the text chunk
    embedding = generate_embeddings(doc["text"])
    doc["embedding"] = embedding

print(doc["embedding"])


# Upsert the chunked documents into the ChromaDB collection
print("===Upserting documents into the collection===")
for doc in chunked_documents:
    print(F"===Upserting document {doc['id']}===")
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]],
        embeddings=[doc["embedding"]]
    )
print("===Documents upserted successfully===")

# Function to query the documents in the collection
def query_documents(question, n_results=2):
    print("===Querying documents===")
    #query_embedding = generate_embeddings(query_text)
    results = collection.query(query_texts=question,
        n_results=n_results
    )
    
    # Extract the relevant chunks
    relevant_chunks = []
    for sublist in results['documents']:
        for doc in sublist:
            relevant_chunks.append(doc)
            print(F"===Found relevant chunk: {doc}===")
    return relevant_chunks
    
# Function to generate a response from the OpenAI model
def generate_response(question, relevant_chunks):
    print("===Generating response from OpenAI===")
    context = "\n\n".join(relevant_chunks)

    prompt = ( "you are a helpful assistant. "
               "Use the following context to answer the question.\n\n"
               f"Context: {context}\n\n"
               f"Question: {question}" )
    print(F"===Prompt: {prompt}===")

    # Generate the response using OpenAI's chat completion
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
    )
    answer = response.choices[0].message.content
    return answer

# Example query
# query_documents("tell me about the world in 10 years acoording what is happening now")

question = " What is databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)
print(F"===Answer: {answer}===")    



