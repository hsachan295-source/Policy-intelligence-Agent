from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def extract_topics(texts, n_topics=3):
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    words = vectorizer.get_feature_names_out()
    topics = []

    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-10:]]
        topics.append((f"Topic {idx+1}", top_words))

    return topics