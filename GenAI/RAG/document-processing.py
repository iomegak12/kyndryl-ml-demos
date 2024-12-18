import os
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


def get_pdf_text(pdf_document):
    text = ""

    pdf_reader = PdfReader(pdf_document)

    for page in pdf_reader.pages:
        text += f"{page.extract_text()}\n"

    return text


def create_documents(pdf_files):
    documents = []

    if pdf_files is not None:
        for pdf_file in pdf_files:
            chunks = get_pdf_text(pdf_file)

            documents.append(
                Document(
                    page_content=chunks,
                    metadata={
                        "source": pdf_file,
                        "type": "PDF",
                        "owner": "Ramkumar JD"
                    }
                )
            )

    return documents


def create_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    return embeddings


def push_documents_to_pinecone(index_name, embeddings, documents):
    if index_name is not None and \
            embeddings is not None and documents is not None:

        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings)

        vector_store.add_documents(documents)

        print("Documents are added to the Vector Index!")


def main():
    try:
        load_dotenv()

        index_name = os.environ["PINECONE_INDEX"]
        directory_path = "./Docs"
        files = os.listdir(directory_path)
        pdf_files = []

        for file in files:
            pdf_file = directory_path + "/" + file
            pdf_files.append(pdf_file)

            print(f"Processing Required ... {pdf_file}")

        documents = create_documents(pdf_files)
        embeddings = create_embeddings()

        push_documents_to_pinecone(index_name, embeddings, documents)

        print("Vector Embeddings are successfully created ...")
    except Exception as error:
        print(f"Error Occurred, Details : {error}")


if __name__ == "__main__":
    main()
