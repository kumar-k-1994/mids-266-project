import torch
import evaluate
import numpy as np

from typing import Type
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)


class DistilbertClassifier:
    def __init__(
        self,
        training_configurations: dict,
        model_choice: str = "distilbert-base-uncased",
    ):
        # Store inputs
        self.training_configurations = training_configurations
        self.model_choice = model_choice

        # Setup Model & Tokenizer
        self.model = DistilBertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_choice,
            num_labels=self.training_configurations["num_labels"],
        )
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_choice)

        # Setup Configurations for Training:
        self._setup_training_arguments()

        # Setup Evaluator:
        self.accuracy = evaluate.load("accuracy")

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.accuracy.compute(predictions=predictions, references=labels)

    def _setup_training_arguments(self):
        self.training_args = TrainingArguments(
            output_dir=self.training_configurations["output_dir"],
            num_train_epochs=self.training_configurations["num_train_epochs"],
            per_device_train_batch_size=self.training_configurations[
                "batch_size_train"
            ],
            per_device_eval_batch_size=self.training_configurations["batch_size_eval"],
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir="./logs",  # directory for storing logs
            logging_steps=100,
            evaluation_strategy="epoch",
        )

    def launch_training(
        self,
        train_dataset: Type[torch.utils.data.Dataset],
        test_dataset: Type[torch.utils.data.Dataset],
    ):
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, padding="max_length"
        )

        self.trainer = Trainer(
            model=self.model,  # the instantiated HF Transformers model to be trained
            args=self.training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=test_dataset,  # evaluation dataset
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()


"""
Sample code to illustrate usage:

> import sys
> sys.path.append('projects/email_understanding/src/')

> from email_dataset import EmailTrainingDataset

> training_configurations = {
>     "output_dir": "checkpoints-v1",
>     "num_train_epochs": 10,
>     "batch_size_train": 16,
>     "batch_size_eval": 64,
> }

> # Initialize Classifier:
> classifier = DistilbertClassifier(
>     training_configurations = training_configurations,
>     model_choice = "distilbert-base-uncased",
> )

> # Use Classifier's Tokenizer:
> my_dataset = EmailTrainingDataset(
>     tokenizer = classifier.tokenizer,
>     label2id = {"invoice": 0, "not_invoice": 1},
>     path2data_train = "training-data-from-gpt.csv",
>     path2data_test = "testing-data-from-gpt.csv",
> )

> # Launch Training:
> classifier.launch_training(
>     train_dataset = my_dataset.train_dataset,
>     test_dataset = my_dataset.test_dataset,
> )
"""
