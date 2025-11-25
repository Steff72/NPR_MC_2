import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import numpy as np
import os
from typing import Dict, Any

class SentimentClassifier:
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 3, output_dir: str = "models"):
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
    def tokenize_function(self, examples):
        return self.tokenizer(examples["sentence"], padding="max_length", truncation=True)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train(self, train_df, val_df, batch_size=16, epochs=3, learning_rate=2e-5):
        # Convert pandas dataframe to HF dataset
        from datasets import Dataset
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # Tokenize
        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_val = val_dataset.map(self.tokenize_function, batched=True)
        
        # Training Arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=10,
            report_to="none" # Disable wandb for now to keep it simple, or enable if configured
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        return trainer

    def evaluate(self, test_df):
        from datasets import Dataset
        test_dataset = Dataset.from_pandas(test_df)
        tokenized_test = test_dataset.map(self.tokenize_function, batched=True)
        
        trainer = Trainer(
            model=self.model,
            compute_metrics=self.compute_metrics
        )
        
        return trainer.evaluate(tokenized_test)

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

if __name__ == "__main__":
    # Simple test
    print("Model class defined.")
