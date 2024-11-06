from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
import numpy as np

# Load the Sentiment140 dataset
dataset = load_dataset("sentiment140")

# Use a smaller subset for training and validation (e.g., 100,000 samples)
sample_size = 100000
small_dataset = dataset['train'].shuffle(seed=42).select(range(sample_size))

# Print the first example to verify the dataset structure
print(dataset['train'][:1])

# Map 'sentiment' labels from [0, 4] to [0, 1]
def map_labels(example):
    example['sentiment'] = int(example['sentiment'] == 4)
    return example

small_dataset = small_dataset.map(map_labels)

# Split the dataset into training and validation sets
train_testvalid = small_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_testvalid['train']
valid_dataset = train_testvalid['test']

# Rename 'sentiment' column to 'label' in both datasets
train_dataset = train_dataset.rename_column('sentiment', 'label')
valid_dataset = valid_dataset.rename_column('sentiment', 'label')

# Define the model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=64)

# Apply the tokenization to the datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_valid = valid_dataset.map(tokenize_function, batched=True)

# Set the format for PyTorch
tokenized_train.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
tokenized_valid.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])

# Verify the format
print(tokenized_train.format['type'])

# Load the pre-trained model with the correct number of labels
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)

# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {'accuracy': accuracy}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")

# Save the model and tokenizer
model_path = './sentiment140-bert-model'
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

# Load the pipeline with GPU utilization
sentiment_pipeline = pipeline(
    'sentiment-analysis',
    model=model_path,
    tokenizer=model_path,
    device=0  # Set to 0 to use the first GPU device
)

# Test sentences
sentences = [
    "I love this product! It's fantastic.",
    "This was the worst experience I've ever had.",
    "I hate this product it is shit",
    "Absolutely amazing service and friendly staff.",
    "I would not recommend this to anyone."
]

# Perform sentiment analysis
for sentence in sentences:
    result = sentiment_pipeline(sentence)[0]
    print(f"Sentence: {sentence}")
    print(f"Sentiment: {result['label']}, Score: {result['score']:.4f}\n")