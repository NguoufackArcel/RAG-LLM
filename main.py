import os
from typing import List
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util

class SimpleRAG:
    def __init__(self, retriever_model_name: str, generator_model_name: str):
        self.retriever = SentenceTransformer(retriever_model_name)
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator_model = AutoModel.from_pretrained(generator_model_name)

    def embed_documents(self, documents: List[str]):
        return self.retriever.encode(documents, convert_to_tensor=True)

    def retrieve(self, query: str, documents: List[str], top_k: int = 3):
        doc_embeddings = self.embed_documents(documents)
        query_embedding = self.retriever.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, doc_embeddings, top_k=top_k)
        return [documents[hit['corpus_id']] for hit in hits[0]]

    def generate_answer(self, query: str, context: str):
        input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
        inputs = self.generator_tokenizer(input_text, return_tensors="pt")
        outputs = self.generator_model.generate(**inputs, max_length=128)
        return self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Example usage
    documents = [
        "The Eiffel Tower is located in Paris.",
        "The Great Wall of China is visible from space.",
        "Python is a popular programming language."
    ]
    query = "Where is the Eiffel Tower?"

    rag = SimpleRAG("all-MiniLM-L6-v2", "gpt2")
    retrieved_docs = rag.retrieve(query, documents)
    context = " ".join(retrieved_docs)
    answer = rag.generate_answer(query, context)
    print("Answer:", answer)