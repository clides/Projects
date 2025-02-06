import praw
from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
from dotenv import load_dotenv

#########################################################
load_dotenv()
client_id = os.getenv('REDDIT_CLIENT_ID')
client_secret = os.getenv('REDDIT_CLIENT_SECRET')
user_agent = os.getenv('REDDIT_USER_AGENT')
#########################################################

# Setting up the Hugging Face DistilBERT model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
# print(nlp("I love this product!"))

# Setting up the Reddit API
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

def check_subreddit_sentiment(subreddit_name, number_of_posts):
    contents = []
    positives = 0
    negatives = 0
    
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.hot(limit=number_of_posts):
        # print(post.selftext)
        if nlp(post.selftext, truncation=True)[0]['label'] == 'POSITIVE':
            positives += 1
        else:
            negatives += 1
            
    return subreddit_name, positives, negatives

subreddit_name, positives, negatives = check_subreddit_sentiment('PTCGP', 20)
print(f"Subreddit: {subreddit_name}, Positives: {positives}, Negatives: {negatives}")