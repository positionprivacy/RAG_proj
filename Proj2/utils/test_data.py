from vector_store import VectorStore


if __name__ == "__main__":
    test_vector_store = VectorStore()
    for collection_name in test_vector_store.get_all_courses():
        print(f"Searching in collection: {collection_name}")
    query = "什么是毛泽东思想的精髓？" 
    results = test_vector_store.search(query=query, top_k=20)

    print(f"Top results for collection '{collection_name}':")
    print(results)