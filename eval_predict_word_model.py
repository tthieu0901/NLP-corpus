#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation Script for Chinese Masked Language Model
Evaluates the model trained by build_predict_word_model.py on Ming History corpus

This script provides multiple evaluation methods:
1. Perplexity calculation on test data
2. Masked word prediction accuracy
3. Semantic coherence evaluation
4. Domain-specific knowledge assessment
5. Comparative analysis with baseline models
"""

import os
import re
import json
import math
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer, BertForMaskedLM, 
    pipeline, DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MingHistoryDataProcessor:
    """Process Ming History text files for evaluation"""
    
    def __init__(self, data_folder: str):
        self.data_folder = Path(data_folder)
        
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
    
    def load_test_texts(self, max_files: int = 50) -> List[str]:
        """Load a subset of texts for evaluation"""
        logger.info(f"Loading test texts from {self.data_folder}")
        
        txt_files = [f for f in self.data_folder.glob("*.txt") 
                    if f.name != "crawl_summary.txt"]
        
        # Take a random sample for evaluation
        random.shuffle(txt_files)
        test_files = txt_files[:max_files]
        
        test_texts = []
        for file_path in test_files:
            text_content = self.extract_text_content(file_path)
            if text_content.strip():
                test_texts.append(text_content.strip())
        
        logger.info(f"Loaded {len(test_texts)} test documents")
        return test_texts

class ModelEvaluator:
    """Comprehensive evaluation of the trained Chinese MLM"""
    
    def __init__(self, model_path: str, baseline_model: str = "bert-base-chinese"):
        self.model_path = Path(model_path)
        self.baseline_model = baseline_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load trained model
        logger.info(f"Loading trained model from {model_path}")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForMaskedLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load baseline model for comparison
        logger.info(f"Loading baseline model: {baseline_model}")
        self.baseline_tokenizer = BertTokenizer.from_pretrained(baseline_model)
        self.baseline_model_obj = BertForMaskedLM.from_pretrained(baseline_model)
        self.baseline_model_obj.to(self.device)
        self.baseline_model_obj.eval()
        
        # Create pipelines
        self.fill_mask = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)
        self.baseline_pipeline = pipeline("fill-mask", model=self.baseline_model_obj, tokenizer=self.baseline_tokenizer, device=0 if torch.cuda.is_available() else -1)
        
        # Evaluation results storage
        self.results = {}
    
    def calculate_perplexity(self, texts: List[str], max_length: int = 128, batch_size: int = 16) -> float:
        """Calculate perplexity on test texts"""
        logger.info("Calculating model perplexity...")
        
        # Prepare data
        examples = []
        for text in texts:
            # Split long texts into chunks
            tokens = self.tokenizer.tokenize(text)
            chunk_size = max_length - 2  # Account for [CLS] and [SEP]
            
            for i in range(0, len(tokens), chunk_size):
                chunk = tokens[i:i + chunk_size]
                if len(chunk) >= 10:  # Only use reasonable chunks
                    chunk_text = self.tokenizer.convert_tokens_to_string(chunk)
                    examples.append({"text": chunk_text})
        
        if not examples:
            logger.error("No valid examples for perplexity calculation")
            return float('inf')
        
        # Create dataset
        dataset = HFDataset.from_list(examples[:1000])  # Limit for faster evaluation
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_special_tokens_mask=True
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        # Data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator
        )
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                # Calculate loss only on masked tokens
                loss = outputs.loss
                total_loss += loss.item()
                total_tokens += 1
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        logger.info(f"Model perplexity: {perplexity:.2f}")
        self.results['perplexity'] = perplexity
        return perplexity
    
    def evaluate_masked_prediction_accuracy(self) -> Dict[str, float]:
        """Evaluate accuracy on domain-specific masked predictions"""
        logger.info("Evaluating masked prediction accuracy...")
        
        # Domain-specific test cases with expected answers
        test_cases = [
            # Historical titles and names
            {"sentence": "太祖起[MASK]州，所至必克。", "expected": ["濠", "淮", "凤"], "category": "地名"},
            {"sentence": "明代[MASK]宦之祸酷矣。", "expected": ["阉", "宦", "权"], "category": "历史概念"},
            {"sentence": "顾时，字时[MASK]，濠人。", "expected": ["举", "望", "用"], "category": "人名字号"},
            
            # Military terms
            {"sentence": "从太祖渡[MASK]，积功由百夫长授元帅。", "expected": ["江", "河", "淮"], "category": "军事地理"},
            {"sentence": "帝念功臣[MASK]苦，特增其禄。", "expected": ["劳", "征", "战"], "category": "军功概念"},
            
            # Administrative terms
            {"sentence": "擢天策卫指挥同[MASK]。", "expected": ["知", "佥", "事"], "category": "官职"},
            {"sentence": "旋师取山[MASK]。", "expected": ["东", "西", "南"], "category": "地理方位"},
            
            # Classical patterns
            {"sentence": "然非诸党人附[MASK]之。", "expected": ["丽", "从", "和"], "category": "古文语法"},
            {"sentence": "迨神宗末[MASK]，讹言朋兴。", "expected": ["年", "世", "时"], "category": "时间概念"},
            {"sentence": "衣冠填于[MASK]犴。", "expected": ["狴", "牢", "狱"], "category": "法律概念"},
            
            # Additional Ming history specific
            {"sentence": "成祖[MASK]北平，起兵靖难。", "expected": ["在", "居", "据"], "category": "历史事件"},
            {"sentence": "洪武[MASK]年，定天下。", "expected": ["元", "二", "三"], "category": "年号"},
            {"sentence": "设六[MASK]，分理庶政。", "expected": ["部", "司", "曹"], "category": "政治制度"},
            {"sentence": "科举取[MASK]，三年一试。", "expected": ["士", "人", "才"], "category": "教育制度"},
            {"sentence": "倭寇侵[MASK]，民不聊生。", "expected": ["边", "境", "海"], "category": "外患"},
        ]
        
        results_by_category = defaultdict(list)
        all_results = []
        
        for i, test_case in enumerate(test_cases):
            sentence = test_case["sentence"]
            expected = test_case["expected"]
            category = test_case["category"]
            
            try:
                # Get predictions from trained model
                predictions = self.fill_mask(sentence)
                top_predictions = [pred['token_str'] for pred in predictions[:5]]
                
                # Check if any expected answer is in top predictions
                hit_found = any(exp in top_predictions for exp in expected)
                
                # Get top prediction details
                if predictions:
                    top_pred = predictions[0]
                    top_token = top_pred['token_str']
                    top_score = top_pred['score']
                    
                    result = {
                        'sentence': sentence,
                        'expected': expected,
                        'predicted': top_token,
                        'score': top_score,
                        'hit': hit_found,
                        'top_5': top_predictions,
                        'category': category
                    }
                    
                    all_results.append(result)
                    results_by_category[category].append(hit_found)
                    
                    logger.info(f"Test {i+1}: {'✓' if hit_found else '✗'} {sentence}")
                    logger.info(f"  Expected: {expected}, Got: {top_token} ({top_score:.3f})")
                
            except Exception as e:
                logger.error(f"Error testing sentence: {sentence}, Error: {e}")
        
        # Calculate accuracy metrics
        overall_accuracy = sum(r['hit'] for r in all_results) / len(all_results) if all_results else 0
        
        category_accuracies = {}
        for category, hits in results_by_category.items():
            category_accuracies[category] = sum(hits) / len(hits) if hits else 0
        
        accuracy_results = {
            'overall_accuracy': overall_accuracy,
            'category_accuracies': category_accuracies,
            'detailed_results': all_results
        }
        
        self.results['prediction_accuracy'] = accuracy_results
        logger.info(f"Overall prediction accuracy: {overall_accuracy:.3f}")
        return accuracy_results
    
    def compare_with_baseline(self) -> Dict[str, float]:
        """Compare performance with baseline BERT model"""
        logger.info("Comparing with baseline model...")
        
        test_sentences = [
            "太祖起[MASK]州，所至必克。",
            "明代[MASK]宦之祸酷矣。",
            "从太祖渡[MASK]，积功由百夫长授元帅。",
            "帝念功臣[MASK]苦，特增其禄。",
            "成祖[MASK]北平，起兵靖难。"
        ]
        
        comparison_results = []
        
        for sentence in test_sentences:
            try:
                # Get predictions from both models
                trained_preds = self.fill_mask(sentence)
                baseline_preds = self.baseline_pipeline(sentence)
                
                if trained_preds and baseline_preds:
                    trained_top = trained_preds[0]
                    baseline_top = baseline_preds[0]
                    
                    comparison_results.append({
                        'sentence': sentence,
                        'trained_prediction': trained_top['token_str'],
                        'trained_score': trained_top['score'],
                        'baseline_prediction': baseline_top['token_str'],
                        'baseline_score': baseline_top['score'],
                        'score_improvement': trained_top['score'] - baseline_top['score']
                    })
                    
            except Exception as e:
                logger.error(f"Error comparing models on: {sentence}, Error: {e}")
        
        # Calculate average score improvement
        avg_improvement = np.mean([r['score_improvement'] for r in comparison_results]) if comparison_results else 0
        
        comparison_summary = {
            'average_score_improvement': avg_improvement,
            'detailed_comparisons': comparison_results
        }
        
        self.results['baseline_comparison'] = comparison_summary
        logger.info(f"Average confidence score improvement: {avg_improvement:.3f}")
        return comparison_summary
    
    def evaluate_semantic_coherence(self, texts: List[str]) -> Dict[str, float]:
        """Evaluate semantic coherence of predictions in context"""
        logger.info("Evaluating semantic coherence...")
        
        coherence_scores = []
        random.shuffle(texts)
        
        for text in texts[:20]:  # Evaluate on 20 random texts
            try:
                sentences = re.split(r'[。！？]', text)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                
                if not sentences:
                    continue
                
                for sentence in sentences[:3]:  # Test up to 3 sentences per text
                    # Find suitable characters to mask
                    chars = list(sentence)
                    mask_candidates = [i for i, char in enumerate(chars) 
                                     if char not in '，。！？：；（）【】""''']
                    
                    if len(mask_candidates) < 3:
                        continue
                    
                    # Mask a random character
                    mask_idx = random.choice(mask_candidates)
                    original_char = chars[mask_idx]
                    
                    # Create masked sentence
                    masked_chars = chars.copy()
                    masked_chars[mask_idx] = '[MASK]'
                    masked_sentence = ''.join(masked_chars)
                    
                    # Get prediction
                    predictions = self.fill_mask(masked_sentence)
                    if predictions:
                        predicted_char = predictions[0]['token_str']
                        score = predictions[0]['score']
                        
                        # Simple coherence check: character match or high confidence
                        coherent = (predicted_char == original_char) or (score > 0.1)
                        coherence_scores.append(1.0 if coherent else 0.0)
                        
            except Exception as e:
                logger.error(f"Error in coherence evaluation: {e}")
                continue
        
        coherence_score = np.mean(coherence_scores) if coherence_scores else 0.0
        
        coherence_results = {
            'coherence_score': coherence_score,
            'num_evaluations': len(coherence_scores)
        }
        
        self.results['semantic_coherence'] = coherence_results
        logger.info(f"Semantic coherence score: {coherence_score:.3f}")
        return coherence_results
    
    def analyze_vocabulary_coverage(self, texts: List[str]) -> Dict[str, any]:
        """Analyze vocabulary coverage and OOV handling"""
        logger.info("Analyzing vocabulary coverage...")
        
        # Collect all characters from texts
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        # Check coverage in tokenizer vocabulary
        vocab = set(self.tokenizer.vocab.keys())
        covered_chars = all_chars.intersection(vocab)
        oov_chars = all_chars - covered_chars
        
        coverage_rate = len(covered_chars) / len(all_chars) if all_chars else 0
        
        vocab_analysis = {
            'total_unique_chars': len(all_chars),
            'covered_chars': len(covered_chars),
            'oov_chars': len(oov_chars),
            'coverage_rate': coverage_rate,
            'sample_oov_chars': list(oov_chars)[:20]
        }
        
        self.results['vocabulary_analysis'] = vocab_analysis
        logger.info(f"Vocabulary coverage: {coverage_rate:.3f}")
        return vocab_analysis
    
    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report"""
        logger.info("Generating evaluation report...")
        
        report = []
        report.append("="*80)
        report.append("CHINESE MASKED LANGUAGE MODEL EVALUATION REPORT")
        report.append("="*80)
        report.append(f"Model Path: {self.model_path}")
        report.append(f"Evaluation Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Perplexity Results
        if 'perplexity' in self.results:
            report.append("1. PERPLEXITY EVALUATION")
            report.append("-" * 30)
            report.append(f"Model Perplexity: {self.results['perplexity']:.2f}")
            if self.results['perplexity'] < 50:
                report.append("✓ Excellent perplexity score")
            elif self.results['perplexity'] < 100:
                report.append("✓ Good perplexity score")
            else:
                report.append("⚠ High perplexity - model may need more training")
            report.append("")
        
        # Prediction Accuracy Results
        if 'prediction_accuracy' in self.results:
            acc_results = self.results['prediction_accuracy']
            report.append("2. MASKED PREDICTION ACCURACY")
            report.append("-" * 30)
            report.append(f"Overall Accuracy: {acc_results['overall_accuracy']:.3f}")
            report.append("\nAccuracy by Category:")
            for category, accuracy in acc_results['category_accuracies'].items():
                report.append(f"  {category}: {accuracy:.3f}")
            report.append("")
        
        # Baseline Comparison
        if 'baseline_comparison' in self.results:
            comp_results = self.results['baseline_comparison']
            report.append("3. BASELINE MODEL COMPARISON")
            report.append("-" * 30)
            report.append(f"Average Score Improvement: {comp_results['average_score_improvement']:.3f}")
            if comp_results['average_score_improvement'] > 0:
                report.append("✓ Model performs better than baseline")
            else:
                report.append("⚠ Model performs worse than baseline")
            report.append("")
        
        # Semantic Coherence
        if 'semantic_coherence' in self.results:
            coh_results = self.results['semantic_coherence']
            report.append("4. SEMANTIC COHERENCE")
            report.append("-" * 30)
            report.append(f"Coherence Score: {coh_results['coherence_score']:.3f}")
            report.append(f"Evaluations: {coh_results['num_evaluations']}")
            report.append("")
        
        # Vocabulary Analysis
        if 'vocabulary_analysis' in self.results:
            vocab_results = self.results['vocabulary_analysis']
            report.append("5. VOCABULARY ANALYSIS")
            report.append("-" * 30)
            report.append(f"Total Unique Characters: {vocab_results['total_unique_chars']}")
            report.append(f"Coverage Rate: {vocab_results['coverage_rate']:.3f}")
            report.append(f"OOV Characters: {vocab_results['oov_chars']}")
            report.append("")
        
        # Recommendations
        report.append("6. RECOMMENDATIONS")
        report.append("-" * 30)
        
        if 'perplexity' in self.results:
            if self.results['perplexity'] > 100:
                report.append("• Consider additional training epochs to reduce perplexity")
            
        if 'prediction_accuracy' in self.results:
            acc = self.results['prediction_accuracy']['overall_accuracy']
            if acc < 0.3:
                report.append("• Model accuracy is low - consider domain-specific fine-tuning")
            elif acc > 0.7:
                report.append("✓ Model shows excellent domain knowledge")
        
        if 'baseline_comparison' in self.results:
            improvement = self.results['baseline_comparison']['average_score_improvement']
            if improvement < 0:
                report.append("• Model underperforms baseline - review training data quality")
        
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_detailed_results(self, output_path: str):
        """Save detailed evaluation results to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results saved to {output_path}")
    
    def create_visualization(self, output_dir: str):
        """Create evaluation visualization plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Plot 1: Accuracy by Category
        if 'prediction_accuracy' in self.results:
            acc_data = self.results['prediction_accuracy']['category_accuracies']
            if acc_data:
                plt.figure(figsize=(12, 6))
                categories = list(acc_data.keys())
                accuracies = list(acc_data.values())
                
                bars = plt.bar(categories, accuracies, color='skyblue', alpha=0.7)
                plt.title('Prediction Accuracy by Category', fontsize=14, fontweight='bold')
                plt.xlabel('Category')
                plt.ylabel('Accuracy')
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 1)
                
                # Add value labels on bars
                for bar, acc in zip(bars, accuracies):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{acc:.2f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(output_dir / 'accuracy_by_category.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Plot 2: Model Comparison
        if 'baseline_comparison' in self.results:
            comp_data = self.results['baseline_comparison']['detailed_comparisons']
            if comp_data:
                plt.figure(figsize=(10, 6))
                sentences = [f"Test {i+1}" for i in range(len(comp_data))]
                trained_scores = [d['trained_score'] for d in comp_data]
                baseline_scores = [d['baseline_score'] for d in comp_data]
                
                x = np.arange(len(sentences))
                width = 0.35
                
                plt.bar(x - width/2, trained_scores, width, label='Trained Model', color='lightcoral', alpha=0.7)
                plt.bar(x + width/2, baseline_scores, width, label='Baseline Model', color='lightblue', alpha=0.7)
                
                plt.title('Model Comparison: Confidence Scores', fontsize=14, fontweight='bold')
                plt.xlabel('Test Cases')
                plt.ylabel('Confidence Score')
                plt.xticks(x, sentences, rotation=45)
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")

def main():
    """Main evaluation pipeline"""
    
    # Configuration
    MODEL_PATH = "chinese_ming_history_mlm"
    DATA_FOLDER = "ming_history_chapters_new"
    OUTPUT_DIR = "evaluation_results"
    BASELINE_MODEL = "bert-base-chinese"
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    logger.info("Starting comprehensive model evaluation...")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Data: {DATA_FOLDER}")
    logger.info(f"Baseline: {BASELINE_MODEL}")
    
    try:
        # Load test data
        data_processor = MingHistoryDataProcessor(DATA_FOLDER)
        test_texts = data_processor.load_test_texts(max_files=100)
        
        if not test_texts:
            logger.error("No test texts loaded. Check data folder.")
            return
        
        # Initialize evaluator
        evaluator = ModelEvaluator(MODEL_PATH, BASELINE_MODEL)
        
        # Run evaluations
        logger.info("Running perplexity calculation...")
        evaluator.calculate_perplexity(test_texts)
        
        logger.info("Running prediction accuracy evaluation...")
        evaluator.evaluate_masked_prediction_accuracy()
        
        logger.info("Running baseline comparison...")
        evaluator.compare_with_baseline()
        
        logger.info("Running semantic coherence evaluation...")
        evaluator.evaluate_semantic_coherence(test_texts)
        
        logger.info("Running vocabulary analysis...")
        evaluator.analyze_vocabulary_coverage(test_texts)
        
        # Generate and save results
        report = evaluator.generate_evaluation_report()
        
        # Save report
        report_path = Path(OUTPUT_DIR) / "evaluation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save detailed results
        evaluator.save_detailed_results(Path(OUTPUT_DIR) / "detailed_results.json")
        
        # Create visualizations
        evaluator.create_visualization(OUTPUT_DIR)
        
        # Print report
        print("\n" + report)
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {OUTPUT_DIR}")
        
        print(f"\n📊 Evaluation Results Summary:")
        print(f"📝 Report: {report_path}")
        print(f"📊 Detailed Results: {Path(OUTPUT_DIR) / 'detailed_results.json'}")
        print(f"📈 Visualizations: {OUTPUT_DIR}/")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    main()
