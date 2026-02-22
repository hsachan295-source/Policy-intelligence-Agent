def agent_router(query, sentiment_summary, topics, rag_func):
    query = query.lower()

    if "sentiment" in query:
        return sentiment_summary
    elif "topic" in query:
        return topics
    else:
        return rag_func(query)