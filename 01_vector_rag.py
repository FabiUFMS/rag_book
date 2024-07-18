from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


def read_pdf(path):
    if path:
        loader = PyPDFLoader(path)
        pages = loader.load()
        return pages
    else:
        print("No path provided, check the path and try again. Make sure the file is a PDF file.")


def split_and_chunk(pages):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, 
                                                chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)
    print("Number of chunks created: ", len(chunks))
    return chunks


def vectorize_chunks(chunks):
    try:
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=OllamaEmbeddings(model="nomic-embed-text", 
                                       show_progress=True),
            collection_name="local_rag_vector_db",
            persist_directory="./local_rag_vector_db"
        )
        print("Vectorization finished successfully.")
    except Exception as e:
        print(f"Failed vectorization {e}")
        vector_db = None
    return vector_db



def main():
    path = "chap1_fire_and_blood.pdf"
    doc = read_pdf(path)
    doc = split_and_chunk(doc)
    doc = vectorize_chunks(doc)
    
    
if __name__ == "__main__":
    main()


