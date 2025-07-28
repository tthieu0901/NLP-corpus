#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Masked Language Model Training for Historical Chinese Text
Based on Ming Dynasty Historical Records

This script builds a Masked Language Model using BERT architecture
for classical Chinese text from the Ming History (明史) corpus.
"""

import os
import re
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import glob

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForMaskedLM, BertConfig,
    AdamW, get_linear_schedule_with_warmup,
    pipeline, DataCollatorForLanguageModeling
)
import numpy as np
from datasets import Dataset as HFDataset
import math

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlm_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MingHistoryDataProcessor:
    """Process Ming History text files for MLM training"""
    
    def __init__(self, data_folder: str):
        self.data_folder = Path(data_folder)
        self.text_content = []
        
    def extract_text_content(self, file_path: str) -> str:
        """Extract main text content from a Ming History file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by the separator line
            parts = content.split('=' * 50)
            if len(parts) >= 2:
                # Get the text content after the metadata
                text_content = parts[1].strip()
                # Remove any remaining metadata patterns
                text_content = re.sub(r'^[^：]*：.*\n', '', text_content, flags=re.MULTILINE)
                return text_content
            return ""
            
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            return ""
    
    def load_all_texts(self) -> List[str]:
        """Load and extract text from all files in the data folder"""
        logger.info(f"Loading texts from {self.data_folder}")
        
        # Get all .txt files except crawl_summary.txt
        txt_files = [f for f in self.data_folder.glob("*.txt") 
                    if f.name != "crawl_summary.txt"]
        
        logger.info(f"Found {len(txt_files)} text files")
        
        all_texts = []
        processed_count = 0
        
        for file_path in txt_files:
            text_content = self.extract_text_content(file_path)
            if text_content.strip():
                all_texts.append(text_content.strip())
                processed_count += 1
                
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} files")
        
        logger.info(f"Successfully loaded {len(all_texts)} text documents")
        return all_texts

class ChineseMLMDataset(Dataset):
    """Dataset for Chinese Masked Language Modeling"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._prepare_examples(texts)
        
    def _prepare_examples(self, texts: List[str]) -> List[str]:
        """Prepare text examples by chunking into fixed-size sequences"""
        logger.info("Preparing text examples...")
        
        # Concatenate all texts
        combined_text = " ".join(texts)
        
        # Tokenize the entire text
        tokens = self.tokenizer.tokenize(combined_text)
        logger.info(f"Total tokens: {len(tokens)}")
        
        # Create chunks of max_length - 2 (for [CLS] and [SEP])
        chunk_size = self.max_length - 2
        examples = []
        
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            if len(chunk) >= 10:  # Only keep chunks with reasonable length
                text = self.tokenizer.convert_tokens_to_string(chunk)
                examples.append(text)
        
        logger.info(f"Created {len(examples)} training examples")
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {"text": self.examples[idx]}

class MLMTrainer:
    """Masked Language Model Trainer for Chinese Text"""
    
    def __init__(self, model_name: str = "bert-base-chinese", output_dir: str = "chinese_mlm_model"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize tokenizer and model
        logger.info(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        
    def tokenize_function(self, examples):
        """Tokenize text examples"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_special_tokens_mask=True
        )
    
    def train(self, texts: List[str], epochs: int = 30, batch_size: int = 32, 
              learning_rate: float = 5e-5, max_length: int = 128):
        """Train the Masked Language Model"""
        
        logger.info("Starting MLM training...")
        
        # Create dataset
        dataset = ChineseMLMDataset(texts, self.tokenizer, max_length)
        
        # Convert to HuggingFace dataset format
        text_data = [{"text": example} for example in dataset.examples]
        hf_dataset = HFDataset.from_list(text_data)
        
        # Tokenize dataset
        tokenized_dataset = hf_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        # Data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator
        )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        total_loss = 0
        step = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1
                step += 1
                
                if step % 100 == 0:
                    avg_loss = total_loss / step
                    perplexity = math.exp(avg_loss)
                    logger.info(f"Step {step}, Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
            
            avg_epoch_loss = epoch_loss / num_batches
            epoch_perplexity = math.exp(avg_epoch_loss)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Perplexity: {epoch_perplexity:.2f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch+1}"
                checkpoint_dir.mkdir(exist_ok=True)
                self.model.save_pretrained(checkpoint_dir)
                self.tokenizer.save_pretrained(checkpoint_dir)
                logger.info(f"Checkpoint saved at epoch {epoch+1}")
        
        # Calculate final perplexity
        final_avg_loss = total_loss / step
        final_perplexity = math.exp(final_avg_loss)
        logger.info(f"Training completed. Final Average Loss: {final_avg_loss:.4f}")
        logger.info(f"Final Perplexity: {final_perplexity:.2f}")
        
        return final_perplexity
    
    def save_model(self):
        """Save the trained model and tokenizer"""
        logger.info(f"Saving model to {self.output_dir}")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info("Model saved successfully!")

def test_model(model_path: str, test_sentences: List[str] = None):
    """Test the trained model with masked sentences"""
    logger.info(f"Testing model from {model_path}")
    
    # Load the fine-tuned model
    fill_mask = pipeline("fill-mask", model=model_path, tokenizer=model_path)
    
    # Default test sentences in classical Chinese
    if test_sentences is None:
        test_sentences = [
            "太祖起[MASK]州，所至必克。",
            "明代[MASK]宦之祸酷矣。",
            "顾时，字时[MASK]，濠人。",
            "从太祖渡[MASK]，积功由百夫长授元帅。",
            "帝念功臣[MASK]苦，特增其禄。"
        ]
    
    print("\n" + "="*50)
    print("MODEL INFERENCE RESULTS")
    print("="*50)
    
    for sentence in test_sentences:
        print(f"\nOriginal: {sentence}")
        try:
            results = fill_mask(sentence)
            print("Predictions:")
            for i, result in enumerate(results[:5]):  # Top 5 predictions
                token = result['token_str']
                score = result['score']
                filled_sentence = result['sequence']
                print(f"  {i+1}. {token} (score: {score:.4f}) -> {filled_sentence}")
        except Exception as e:
            print(f"Error predicting: {e}")

def main():
    """Main training pipeline"""
    
    # Configuration
    DATA_FOLDER = "ming_history_chapters"
    MODEL_NAME = "bert-base-chinese"
    OUTPUT_DIR = "chinese_ming_history_mlm"
    EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-5
    MAX_LENGTH = 128
    
    logger.info("Starting Ming History Masked Language Model Training")
    logger.info(f"Data folder: {DATA_FOLDER}")
    logger.info(f"Base model: {MODEL_NAME}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    
    # Check environment
    import sys
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    try:
        # Step 1: Load and process data
        logger.info("Step 1: Loading and processing data...")
        data_processor = MingHistoryDataProcessor(DATA_FOLDER)
        texts = data_processor.load_all_texts()
        
        if not texts:
            logger.error("No texts loaded. Please check the data folder.")
            logger.error(f"Make sure the folder '{DATA_FOLDER}' exists and contains .txt files")
            return
        
        # Statistics
        total_chars = sum(len(text) for text in texts)
        avg_chars = total_chars // len(texts)
        logger.info(f"Loaded {len(texts)} documents with {total_chars:,} total characters")
        logger.info(f"Average characters per document: {avg_chars:,}")
        
        # Step 2: Initialize trainer
        logger.info("Step 2: Initializing trainer...")
        trainer = MLMTrainer(MODEL_NAME, OUTPUT_DIR)
        
        # Step 3: Train the model
        logger.info("Step 3: Starting model training...")
        logger.info("This may take several hours depending on your hardware...")
        
        final_perplexity = trainer.train(
            texts=texts,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            max_length=MAX_LENGTH
        )
        
        # Step 4: Save the final model
        logger.info("Step 4: Saving final model...")
        trainer.save_model()
        
        # Step 5: Test the model
        logger.info("Step 5: Testing the trained model...")
        test_model(OUTPUT_DIR)
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Final model perplexity: {final_perplexity:.2f}")
        logger.info(f"Model saved to: {OUTPUT_DIR}")
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        print(f"✓ Model trained for {EPOCHS} epochs")
        print(f"✓ Final perplexity: {final_perplexity:.2f}")
        print(f"✓ Model saved to: {OUTPUT_DIR}")
        print(f"✓ Processed {len(texts)} documents")
        print("\nNext steps:")
        print("1. Run 'python test_model.py' for interactive testing")
        print("2. Use the model in your applications")
        print("3. Fine-tune further if needed")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("Stack trace:", exc_info=True)
        print(f"\n❌ Training failed: {e}")
        print("Check the log file 'mlm_training.log' for detailed error information")
        raise

if __name__ == "__main__":
    main()
