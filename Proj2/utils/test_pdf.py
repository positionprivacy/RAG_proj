import PyPDF2
from document_loader import DocumentLoader


if __name__ == "__main__":
    test_loader = DocumentLoader(data_dir="D:\\Code\\RAG_proj\\Proj2\\data")
    result = test_loader.load_document(file_path="D:\Code\RAG_proj\Proj2\data\【清晰版】概论课教材（2021年）.pdf")
    print(f"Loaded {len(result)} pages from the PDF document.")
    for i, page in enumerate(result):
        print(f"Page {i + 1} content:\n{page}\n")

        if i == 10:
            break  # Print only the first 10 pages for brevity