import argparse
import os
from projects.pipeline.basic_rag_pipeline import BasicRAGPipeline
from projects.pipeline.multi_modal_rag_pipeline import MultiModalRAGPipeline
from projects.pipeline.langgraph_rag_pipeline import LangGraphRAGPipeline
from shared.utils.chroma_utils import list_existing_collections, delete_collection
from shared.configs.static import RAG_TYPES, DATA_DIR_MAP
from projects.pipeline.rag_ubac_pipeline import RAGUBACPipeline
from projects.pipeline.cache_rag_pipeline import CacheRAGPipeline

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument(
        "--rag_type",
        default="basic-rag",
        choices=RAG_TYPES,
        help="RAG pipeline to use",
    )
    parser.add_argument("-v", "--vectorize", action="store_true", help="(Re-)vectorize data")
    parser.add_argument("--list-collections", action="store_true", help="List collections and exit")
    parser.add_argument("--delete-collection", action="store_true", help="Delete the collection for the specified RAG type and exit")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache collection (only for cache-rag)")
    parser.add_argument("--info", action="store_true", help="Show pipeline and collection information")
    args = parser.parse_args()

    if args.list_collections:
        print("Listing collections in chroma_db")
        try:
            cols = list_existing_collections()
            if cols:
                for c in cols: print(f"  - {c}")
            else:
                print("No collections found.")
        except Exception as e:
            print(f"Error listing collections: {e}")
        return

    if args.delete_collection:
        collection_name = f"{args.rag_type.replace('-', '_')}_collection"
        confirm = input(f"Are you sure you want to delete collection '{collection_name}'? (yes/no): ")
        if confirm.lower() == 'yes' or confirm.lower() == 'y':
            delete_collection(collection_name)
        return

    if args.clear_cache:
        if args.rag_type != "cache-rag":
            print("Error: --clear-cache can only be used with --rag_type cache-rag")
            return
        confirm = input("Are you sure you want to clear the cache collection? (yes/no): ")
        if confirm.lower() == 'yes' or confirm.lower() == 'y':
            data_dir = DATA_DIR_MAP[args.rag_type]
            rag = CacheRAGPipeline(data_dir)
            rag.retriever.clear_cache()
        return


    data_dir = DATA_DIR_MAP[args.rag_type]
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist!")
        return

    if args.rag_type == "basic-rag":
        rag = BasicRAGPipeline(data_dir)
    elif args.rag_type == "multi-modal":
        rag = MultiModalRAGPipeline(data_dir)
    elif args.rag_type == "rag-ubac":
        rag = RAGUBACPipeline(data_dir)
    elif args.rag_type == "cache-rag":
        rag = CacheRAGPipeline(data_dir)
    else:
        rag = LangGraphRAGPipeline(data_dir)

    if args.info:
        info = rag.get_pipeline_info()
        for k, v in info.items():
            print(f"{k}: {v}")
        return

    if args.vectorize:
        print("Vectorizing data...")
        rag.retriever.index_pdfs()

    print(f"{args.rag_type} RAG ready. Type your question or '/exit' or '/quit' to quit.")
    while True:
        q = input("Ask a question: ")
        if q.lower() in ("/exit", "/quit"):
            break
        print(f"Answer: {rag.answer(q)}\n")

if __name__ == "__main__":
    main()