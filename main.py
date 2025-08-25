import argparse
import os
from projects.pipeline.basic_rag_pipeline import BasicRAGPipeline
from projects.pipeline.multi_modal_rag_pipeline import MultiModalRAGPipeline
from shared.utils.chroma_utils import list_existing_collections, delete_collection

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument(
        "rag_type",
        choices=["all", "basic-rag", "multi-modal"],  
        help="Type of RAG pipeline to use (all, basic-rag, or multi-modal)"
    )
    parser.add_argument(
        "-v", "--vectorize",
        action="store_true",
        help="If set, (re-)vectorize the data; otherwise, use existing vector store"
    )
    parser.add_argument(
        "--list-collections",
        action="store_true",
        help="List all existing collections and exit"
    )
    parser.add_argument(
        "--delete-collection",
        action="store_true",
        help="Delete the collection for the specified RAG type and exit"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show pipeline and collection information"
    )
    
    args = parser.parse_args()

    # Handle special commands first
    if args.list_collections:
        print("Listing collections in chroma_db")
        try:
            collections = list_existing_collections()
            if collections:
                print("Existing collections:")
                for col in collections:
                    print(f"  - {col}")
            else:
                print("No collections found.")
        except Exception as e:
            print(f"Error listing collections: {e}")
            # migrate_old_chroma_data()
        return

    if args.delete_collection:
        collection_name = f"{args.rag_type.replace('-', '_')}_collection"
        confirm = input(f"Are you sure you want to delete collection '{collection_name}'? (yes/no): ")
        if confirm.lower() == 'yes':
            delete_collection(collection_name)
        return

    # Handle "all" option
    if args.rag_type == "all":
        print("Running all RAG pipelines...")
        # You can implement logic to run multiple pipelines here
        print("This feature is not yet implemented. Please specify a specific RAG type.")
        return

    # Set data directory based on RAG type
    data_dir = f"data/source_data/{args.rag_type}"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist!")
        print("Please create the directory and add your data files.")
        return

    if args.rag_type == "basic-rag":
        rag = BasicRAGPipeline(data_dir, rag_type=args.rag_type)
        
        if args.info:
            info = rag.get_pipeline_info()
            print(f"Pipeline Info:")
            print(f"  RAG Type: {info['rag_type']}")
            print(f"  Collection: {info['collection_name']}")
            print(f"  Documents: {info['document_count']}")
            return
        
        if args.vectorize:
            print("Vectorizing data...")
            rag.retriever.index_pdfs()
        
        print("Basic RAG system ready. Type your question or 'exit' to quit.")
        while True:
            query = input("Ask a question: ")
            if query.lower() == "exit":
                break
            answer = rag.answer(query)
            print(f"Answer: {answer}\n")
    
    elif args.rag_type == "multi-modal":
        rag = MultiModalRAGPipeline(data_dir)
        
        if args.info:
            info = rag.get_pipeline_info()
            print(f"Multi-Modal Pipeline Info:")
            print(f"  RAG Type: {info['rag_type']}")
            print(f"  Collection: {info['collection_name']}")
            print(f"  Total Documents: {info['document_count']}")
            print(f"  Text Documents: {info['text_documents']}")
            print(f"  Image Documents: {info['image_documents']}")
            print(f"  Vector Store Initialized: {info['vector_store_initialized']}")
            print(f"  Data Directory: {info['data_directory']}")
            return
        
        if args.vectorize:
            print("Vectorizing multi-modal data...")
            rag.retriever.index_pdfs()
        
        print("Multi-Modal RAG system ready. Type your question or 'exit' to quit.")
        while True:
            query = input("Ask a question: ")
            if query.lower() == "exit":
                break
            answer = rag.answer(query)
            print(f"Answer: {answer}\n")
    
    else:
        print(f"RAG type '{args.rag_type}' is not implemented yet.")

if __name__ == "__main__":
    main()
