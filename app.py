import matplotlib
matplotlib.use("Agg")  # Set the backend to Agg for non-GUI environments
import os
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from collect import get_posts
from io import BytesIO
import base64
import numpy as np
import pandas as pd
from datetime import datetime
from wordcloud import WordCloud
import nltk

app = Flask(__name__)
port = int(os.getenv("PORT", 5000))
# Load the model and tokenizer
model_path = './sentiment140-bert-model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create a pipeline for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True, max_length=512)


def get_brand_sentences(text, brand_name):
    """Extracts sentences that mention the brand."""
    sentences = nltk.sent_tokenize(text)
    brand_sentences = [sentence for sentence in sentences if brand_name.lower() in sentence.lower()]
    return ' '.join(brand_sentences)


def get_sentiment(text, brand_name):
    brand_text = get_brand_sentences(text, brand_name)
    if not brand_text:
        # If the brand is not mentioned in the text, return neutral sentiment
        return {'label': 'NEUTRAL', 'score': 0.0}
    """Analyzes the sentiment of a given text."""
    result = sentiment_pipeline(brand_text, truncation=True, max_length=512)[0]
    return {'label': result['label'], 'score': result['score']}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    brand_name = request.form.get('brand_name')
    tweets = get_posts(brand_name)
    
    analyzed_tweets = []
    weighted_sentiment_sum = 0
    total_votes = 0
    sentiments_over_time = []
    text_content = ""

    for tweet in tweets:

        sentiment = get_sentiment(tweet["text"], brand_name)
        sentiment_label = sentiment['label']
        sentiment_score = -sentiment['score'] if sentiment_label == 'LABEL_0' else sentiment['score']
        label_mapping = {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'POSITIVE', 'NEUTRAL': 'NEUTRAL'}
        mapped_label = label_mapping.get(sentiment_label, 'NEUTRAL')        
        tweet_vote = tweet["vote"]
        weighted_sentiment = sentiment_score * tweet_vote
        weighted_sentiment_sum += weighted_sentiment
        total_votes += abs(tweet_vote)

        # Append data for scatter plot and word cloud
        sentiments_over_time.append({
            "date": datetime.strptime(tweet["date"], "%Y-%m-%d %H:%M:%S"),
            "sentiment": sentiment_score,
            "sentiment_label": mapped_label,
            "vote": tweet_vote
        })
        text_content += " " + tweet["text"]
        
        # Store tweet analysis
        analyzed_tweets.append({
            'text': tweet["text"],
            'date': tweet["date"],
            'vote': tweet_vote,
            'sentiment_label': mapped_label,
            'sentiment_score': int(round(sentiment['score'] * 100))  # Convert to integer percentage
        })

    # Calculate the overall weighted sentiment score
    overall_sentiment = weighted_sentiment_sum / total_votes if total_votes != 0 else 0

    # Generate the scatter plot
    sentiment_df = pd.DataFrame(sentiments_over_time)
    sentiment_df.sort_values("date", inplace=True)
    
    # Map sentiment labels to colors
    sentiment_df['color'] = sentiment_df['sentiment_label'].map({'NEGATIVE': '#FF0000', 'POSITIVE': '#0000FF', 'NEUTRAL': '#808080'})

    # Scale the vote for marker size using logarithmic scaling
    sentiment_df['abs_vote'] = sentiment_df['vote'].abs()
    sentiment_df['log_vote'] = np.log(sentiment_df['abs_vote'] + 1)

    # Normalize the log_vote to a range between min_size and max_size
    min_size = 20
    max_size = 200
    min_log_vote = sentiment_df['log_vote'].min()
    max_log_vote = sentiment_df['log_vote'].max()
    if max_log_vote - min_log_vote == 0:
        sentiment_df['size'] = min_size
    else:
        sentiment_df['size'] = min_size + (sentiment_df['log_vote'] - min_log_vote) * (max_size - min_size) / (max_log_vote - min_log_vote)
    
    # Ensure minimum marker size
    sentiment_df['size'] = sentiment_df['size'].clip(lower=min_size)

    fig, ax = plt.subplots(figsize=(10, 5))  # Using the non-GUI Agg backend
    scatter = ax.scatter(
        sentiment_df["date"],
        sentiment_df["sentiment"],
        c=sentiment_df["color"],
        s=sentiment_df["size"],
        alpha=0.6
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment Score")
    ax.set_title("Sentiment Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot to a string buffer for rendering
    img_buf = BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    plot_url = base64.b64encode(img_buf.getvalue()).decode()
    plt.close(fig)  # Close the figure after saving to buffer

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_content)
    wc_img_buf = BytesIO()
    wordcloud.to_image().save(wc_img_buf, format='PNG')
    wc_img_buf.seek(0)
    wc_plot_url = base64.b64encode(wc_img_buf.getvalue()).decode()

    # Determine sentiment color and label
    if -0.2 <= overall_sentiment <= 0.2:
        overall_sentiment_label = 'NEUTRAL'
        sentiment_color = '#808080'  # Gray
    elif overall_sentiment > 0.2:
        overall_sentiment_label = 'POSITIVE'
        sentiment_color = '#0000FF'  # Blue
    else:
        overall_sentiment_label = 'NEGATIVE'
        sentiment_color = '#FF0000'  # Red

    return render_template(
        'results.html',
        tweets=analyzed_tweets,
        overall_sentiment_label=overall_sentiment_label,
        overall_sentiment_score=overall_sentiment,
        overall_sentiment_percentage=round((overall_sentiment) * 100)/100,
        sentiment_color=sentiment_color,
        plot_url=plot_url,
        wc_plot_url=wc_plot_url
    )
