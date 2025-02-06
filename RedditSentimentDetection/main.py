import praw
from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Setting up the Hugging Face DistilBERT model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
# print(nlp("I love this product!"))

# Setting up the Reddit API
reddit = praw.Reddit(
    client_id='YOUR_CLIENT_ID',
)