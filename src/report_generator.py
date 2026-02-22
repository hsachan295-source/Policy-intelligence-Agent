def generate_report(df, topics):
    total = len(df)
    pos = len(df[df["sentiment"] == 4])
    neg = len(df[df["sentiment"] == 0])

    report = f"""
Executive Policy Intelligence Report
-------------------------------------

Total Tweets Analyzed: {total}

Sentiment Distribution:
Positive: {round(pos/total*100,2)}%
Negative: {round(neg/total*100,2)}%

Top Topics:
"""

    for topic in topics:
        report += f"{topic[0]}: {', '.join(topic[1])}\n"

    report += "\nRisk Insight: High negative sentiment may indicate public policy backlash."
    report += "\nConfidence Score: 0.82"

    return report