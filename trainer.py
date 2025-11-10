import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate # Used for loading accuracy metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import os
import datasets # Used for cache control

# Configurations
# The cleaned data file with embeddings
CLEANED_DATA_FILE = 'knowledge_base_with_embeddings.csv' 
# The column in CSV should containing the ticket text
TEXT_COLUMN = 'Full_Ticket_Text'  
# The column in CSV containing the target team name
TARGET_TEAM_COLUMN = 'Assigned Team' 

MODEL_NAME = 'distilbert-base-uncased' # Base model for fine-tuning
SAVE_DIR_TEAM = './fine_tuned_model_team' # The required output directory for app.py
NUM_EPOCHS = 5

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
accuracy_metric = evaluate.load("accuracy")

#Helper Functions

def tokenize_function(examples):
    """Tokenizes the text column."""
    # Uses the renamed column 'text_input' from the main function
    return tokenizer(examples['text_input'], truncation=True, padding="max_length", max_length=128)

def compute_metrics(eval_pred):
    """Computes accuracy during evaluation."""
    preds = np.argmax(eval_pred.predictions, axis=1)
    return accuracy_metric.compute(predictions=preds, references=eval_pred.label_ids)

# Main Training 

def main():
    try:
        # Load local CSV data
        print(f"Loading local data from {CLEANED_DATA_FILE}...")
        df = pd.read_csv(CLEANED_DATA_FILE, sep='|') 
        
        # 1.Prepare Data

        df = df[[TEXT_COLUMN, TARGET_TEAM_COLUMN]].dropna()
        
        # 2. Map Labels: Convert team names (strings) to unique integer labels
        unique_labels = df[TARGET_TEAM_COLUMN].unique()
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        id_to_label = {i: label for i, label in enumerate(unique_labels)}
        df['labels'] = df[TARGET_TEAM_COLUMN].map(label_to_id)
        
        # Rename text column for consistency
        df = df.rename(columns={TEXT_COLUMN: 'text_input'}) 
        
        # 3.Split the data
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'])

        # Convert to Hugging Face Dataset format # Rename the text column to 'text_input' for consistent use with the tokenizer
        train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
        val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
        
        print(f"Data loaded. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
    except FileNotFoundError:
        print(f"FATAL ERROR: Data file not found at {CLEANED_DATA_FILE}. Please check the file path.")
        return
    except KeyError as e:
        print(f"FATAL ERROR: Required column {e} not found in CSV. Please ensure you use the exact names: '{TEXT_COLUMN}' and '{TARGET_TEAM_COLUMN}'.")
        return
    except Exception as e:
        print(f"FATAL ERROR loading data: {e}")
        return

    # 4. Tokenization
    tokenized_train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=['text_input', TARGET_TEAM_COLUMN] 
    )
    tokenized_val_dataset = val_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=['text_input', TARGET_TEAM_COLUMN] 
    )

    # 5. Model Initialization
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(unique_labels),
        id2label=id_to_label,
        label2id=label_to_id
    )

    # 6. Define Training Parameters and Trainer
    training_args = TrainingArguments(
        output_dir="./team_model_results",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",  
        save_strategy="epoch",
        logging_dir='./team_logs',
        logging_steps=100,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 7. Train and Save the Model
    print("Starting training for Assigned Team model...")
    trainer.train()

    print(f"Saving model to {SAVE_DIR_TEAM}...")
    trainer.save_model(SAVE_DIR_TEAM)
    tokenizer.save_pretrained(SAVE_DIR_TEAM)
    print(f"Model and tokenizer saved successfully to {SAVE_DIR_TEAM}.")


if __name__ == "__main__":
    datasets.disable_caching()
    main()