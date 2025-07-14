from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

def load_pdf(data_dir):
    loader = DirectoryLoader(
        path=data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def convert_data():
    extracted_documents = load_pdf(r"C:\Advance_Projects\AI-Trip-Planner\data")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    text_chunks = splitter.split_documents(extracted_documents)
    return text_chunks

if __name__ == "__main__":
    chunks = convert_data()
    print(f"Total chunks created: {len(chunks)}\n")
    for i, chunk in enumerate(chunks[:5]):
        print(f"Chunk {i+1}:")
        print(chunk.page_content[:300], "\n---")
