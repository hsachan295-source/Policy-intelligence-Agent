from src.data_loader import load_data
from src.preprocessing import clean_text
from src.sentiment_model import train_sentiment_model
from src.topic_model import extract_topics
from src.embeddings import generate_embeddings
from src.vector_store import VectorStore
from src.rag_pipeline import rag_answer
from src.agent import agent_router
from src.report_generator import generate_report


if __name__ == "__main__":
    file_path = "data/sentiment140.csv"
    keyword = "love"   # changed keyword

    df = load_data(file_path, keyword=keyword)

    print("Total tweets after keyword filter:", len(df))

    # Preprocess
    df["text"] = df["text"].apply(clean_text)

    # Sentiment model
    model, vectorizer = train_sentiment_model(df)

    sentiment_summary = "Sentiment model trained successfully."

    # Topics
    topics = extract_topics(df["text"].tolist())

    # Embeddings
    embeddings = generate_embeddings(df["text"].tolist())
    vector_store = VectorStore(embeddings)

    # RAG
    def rag_func(q):
        return rag_answer(q, vector_store, df["text"].tolist())

    # Generate Report
    report = generate_report(df, topics)
    print(report)

    # Agent Example
    user_query = "Give sentiment summary"
    response = agent_router(user_query, sentiment_summary, topics, rag_func)
    print("\nAgent Response:\n", response)