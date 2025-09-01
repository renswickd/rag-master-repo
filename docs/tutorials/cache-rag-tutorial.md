# Cache-RAG Tutorial

This tutorial shows how to use the Cache-RAG pipeline that implements intelligent caching with LangGraph orchestration.

## 1. What is Cache-RAG?

Cache-RAG is a RAG pipeline that maintains a cache of previous question-answer pairs to improve response times and reduce redundant processing. It uses two separate collections:

- **Retriever Collection** (`cache_rag_collection`): Stores document chunks for RAG retrieval
- **Cache Collection** (`cache_rag_cache_collection`): Stores cached question-answer pairs

## 2. How it works

The pipeline uses LangGraph to orchestrate the following flow:

1. **Check Cache**: First checks if a similar question has been asked before (75% similarity threshold)
2. **Cache Hit**: If similarity ≥ 50%, returns the cached answer immediately
3. **Cache Miss**: If similarity < 50%, proceeds to RAG retrieval
4. **RAG Retrieval**: Retrieves relevant documents from the retriever collection
5. **Generate Answer**: Uses LLM to generate answer from retrieved context
6. **Write Cache**: Stores the new question-answer pair in cache *(only if it's a real answer)*

## 3. Key Features

- **Smart Caching**: Only caches actual answers, not default "no content" responses
- **Independent Collections**: Separate collections for documents and cache
- **LangGraph Orchestration**: Conditional flow based on cache hits
- **Cache Management**: Built-in cache clearing functionality

## 4. Setup and Usage

### Index Documents

Place your PDF documents in `data/source_data/basic-rag/` (reuses basic-rag data), then run:

```bash
python main.py --rag_type cache-rag --vectorize
```

This creates the `cache_rag_collection` with your document chunks.

### Query with Caching

```bash
python main.py --rag_type cache-rag
```

**First query**: Goes through full RAG pipeline and caches the result
**Subsequent identical/similar queries**: Returns cached result immediately

### Cache Management

Clear the cache collection:
```bash
python main.py --rag_type cache-rag --clear-cache
```

View collection information:
```bash
python main.py --rag_type cache-rag --info
```

## 5. Example Flow

```
User: "What is attention mechanism?"

First time:
1. Check cache → No hit
2. Retrieve documents → Find relevant chunks
3. Generate answer → "Attention mechanism is..."
4. Cache answer → Store for future use

Second time (same/similar question):
1. Check cache → Hit found!
2. Return cached answer immediately
```

## 6. Configuration

Edit `shared/configs/static.py`:

```python
CACHE_RAG_TYPE = "cache-rag"
DEFAULT_NO_CONTENT_MESSAGE = "I am a helpful assistant..."
CACHE_SIMILARITY_THRESHOLD = 0.5  # 50% similarity threshold for cache hits
```

- `DEFAULT_NO_CONTENT_MESSAGE`: Used to identify responses that shouldn't be cached
- `CACHE_SIMILARITY_THRESHOLD`: Minimum similarity score (0.75 = 75%) required for cache hits

## 7. Technical Details

### Files Structure
- `projects/pipeline/cache_rag_pipeline.py`: LangGraph orchestration
- `projects/retriever/cache_rag_retriever.py`: Dual collection management
- Uses `BASIC_RAG_PROMPT` for consistency with other pipelines

### LangGraph Flow
```
check_cache → [cache_hit?] → END (if hit)
            ↓
          retrieve → generate → write_cache → END
```

### Metadata Structure
- **Retriever collection**: `{"source": "cache-rag", "type": "retriever"}`
- **Cache collection**: `{"type": "cache", "question": "original_question"}`

## 8. Benefits

- **Performance**: Instant responses for repeated questions
- **Cost Efficiency**: Reduces LLM API calls for cached queries
- **Consistency**: Same questions get identical answers
- **Smart Filtering**: Only meaningful answers are cached

## 9. Maintenance

### Regular Cache Clearing
Periodically clear cache to ensure fresh answers:
```bash
python main.py --rag_type cache-rag --clear-cache
```

### Monitoring
Check cache effectiveness:
```bash
python main.py --rag_type cache-rag --info
```

Look at `cache_count` vs `retriever_count` to understand cache usage.

## 10. Troubleshooting

- **Cache not working**: Ensure questions are similar enough for vector similarity
- **Wrong answers cached**: Clear cache and re-query after document updates
- **Performance issues**: Monitor cache collection size and clear periodically

## 11. Advanced Usage

### Custom Similarity Threshold
Adjust the similarity threshold in `shared/configs/static.py`:

```python
CACHE_SIMILARITY_THRESHOLD = 0.8  # Increase to 80% for stricter matching
```

Or pass it dynamically:
```python
pipeline.answer("Your question", similarity_threshold=0.8)
```

### Cache Expiration
Add timestamp metadata to implement TTL cache expiration:

```python
"timestamp": datetime.now().isoformat()
```
