import argparse
from projects.pipeline.basic_rag_pipeline import BasicRAGPipeline

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument(
        "rag_type",
        choices=["basic-rag"],  
        help="Type of RAG pipeline to use (currently only 'basic-rag' is supported)"
    )
    parser.add_argument(
        "-v", "--vectorize",
        action="store_true",
        help="If set, (re-)vectorize the data; otherwise, use existing vector store"
    )
    args = parser.parse_args()

    data_dir = None
    rag = None

    if args.rag_type == "basic-rag":
        data_dir = "data/source_data/basic-rag/"
        rag = BasicRAGPipeline(data_dir)
        
    else:
        print(f"RAG type '{args.rag_type}' is not implemented yet.")

    if args.vectorize:
        print("Vectorizing data...")
        rag.retriever.index_pdfs()
    print("RAG system ready. Type your question or '/exit' or '/quit' to quit.")
    while True:
        query = input("Ask a question: ")
        if query.lower() == "/exit" or query.lower() == "/quit":
            break
        answer = rag.answer(query)
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()
