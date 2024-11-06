from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Specify the path where the model and tokenizer were saved
model_path = './sentiment140-bert-model'

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create a pipeline for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Test sentences
test_sentences = [
    "I love this product! It's fantastic.",
    "This was the worst experience I've ever had.",
    "I hate this product it is shit",
    "Absolutely amazing service and friendly staff.",
    "I would not recommend this to anyone."
]

# Get predictions
results = sentiment_pipeline(test_sentences)

# Display results
for sentence, result in zip(test_sentences, results):
    print(f"Sentence: {sentence}")
    print(f"Sentiment: {result['label']}, Score: {result['score']:.4f}\n")