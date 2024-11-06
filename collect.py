import praw
import os
from datetime import datetime
import os

# Set up Reddit API credentials
reddit = praw.Reddit(
    client_id= os.getenv("REDDIT_CLIENT_ID"),
    client_secret= os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent= os.getenv("REDDIT_USER_AGENT")
)

def get_posts(query, limit=100):
    """
    Fetches Reddit posts for a specific query using PRAW.
    
    Parameters:
        query (str): The search term to query on Reddit.
        limit (int): The maximum number of posts to retrieve. Default is 10.

    Returns:
        list of dict: A list of dictionaries, each containing 'text', 'date', and 'score' of a Reddit post.
    """
    posts = []
    for submission in reddit.subreddit('all').search(query, limit=limit):
        post = {
            'text': submission.title + ' ' + (submission.selftext or ''),
            'date': datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            'vote': submission.score
        }
        posts.append(post)
    
    return posts