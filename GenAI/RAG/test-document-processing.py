import os
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


def create_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    return embeddings


def search_similar_documents(query, no_of_documents, index_name, embeddings):
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings)

    similar_documents = vector_store.similarity_search(
        query, k=no_of_documents)

    return similar_documents


def main():
    try:
        load_dotenv()

        index_name = os.environ["PINECONE_INDEX"]
        embeddings = create_embeddings()
        no_of_documents = 2

        query = """
            Experienced Candidates with Embedded Systems 
            
            Requirements:
            
                Bachelors Degree in Computer Science
                At least 5+ years of Experience in Embedded Systems
                Understanding Computer Architecture, Programming Languages and Interfacing Technologies
        """

        search_results = search_similar_documents(
            query, no_of_documents, index_name, embeddings)

        for document_index in range(len(search_results)):
            document = search_results[document_index]

            print(f"Metadata : (Source) : {document.metadata["source"]}")
            print(f"Metadata : (Owner) : {document.metadata["owner"]}")
            print(document.page_content)
    except Exception as error:
        print(f"Error Occurred, Details : {error}")


if __name__ == "__main__":
    main()