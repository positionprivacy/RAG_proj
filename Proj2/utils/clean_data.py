from vector_store import VectorStore

if __name__ == "__main__":
    file_name = "【清晰版】概论课教材（2021年）.pdf"

    test_vector_store = VectorStore()
    
    test_vector_store.delete_documents_by_filename(filename=file_name)