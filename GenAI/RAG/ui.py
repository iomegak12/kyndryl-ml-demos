import os
from dotenv import load_dotenv
import streamlit as st

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain


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


def get_summary_from_llm(current_document):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-0125",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.5,
        max_tokens=1000
    )

    chain = load_summarize_chain(llm, chain_type="stuff")
    summary = chain.run([current_document])

    return summary


def main():
    try:
        load_dotenv()

        index_name = os.environ["PINECONE_INDEX"]
        embeddings = create_embeddings()

        st.set_page_config(page_title="Resume Screening Assistant")
        st.title("Resume Screen AI Assistant")
        st.subheader(
            "This AI Assistant shall help HR Executives to be able to filter resumes that are stored in the database")

        job_description = st.text_area(
            "Paste your JD here ...",
            height=200
        )
        document_count = st.text_input("No. of Resume(s) To Return")
        submit = st.button("Analyze")

        if submit:
            relevant_documents = search_similar_documents(job_description, int(document_count),
                                                          index_name, embeddings)

            for document_index in range(len(relevant_documents)):
                st.subheader(":sparkles:" + str(document_index+1))

                file_name = "** FILE ** " + \
                    relevant_documents[document_index].metadata["source"]

                st.write(file_name)

                with st.expander("Summary of the Resume ..."):
                    summary = get_summary_from_llm(
                        relevant_documents[document_index])

                    st.write(" **** SUMMARY **** \n" + summary)
    except Exception as error:
        print(f"Error Occurred, Details : {error}")


if __name__ == "__main__":
    main()
