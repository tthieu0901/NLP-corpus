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

Usage:
    Standalone: python eval_predict_word_model.py
    Notebook: from eval_predict_word_model import run_notebook_evaluation
              evaluator = run_notebook_evaluation()
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

# Check if running in notebook environment for automatic inline display
try:
    from IPython.display import display, clear_output
    import matplotlib
    if 'ipykernel' in str(type(get_ipython())):
        matplotlib.use('inline')
        NOTEBOOK_MODE = True
        logger.info("Notebook environment detected - will display plots inline")
    else:
        NOTEBOOK_MODE = False
except:
    NOTEBOOK_MODE = False

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
        
        # Domain-specific test cases with expected answers - Expanded set
        test_cases = [
            # Historical titles and names
            {"sentence": "太祖起[MASK]州，所至必克。", "expected": ["濠", "淮", "凤"], "category": "地名"},
            {"sentence": "明代[MASK]宦之祸酷矣。", "expected": ["阉", "宦", "权"], "category": "历史概念"},
            {"sentence": "顾时，字时[MASK]，濠人。", "expected": ["举", "望", "用"], "category": "人名字号"},
            {"sentence": "帝都在[MASK]京。", "expected": ["南", "北", "应"], "category": "地名"},
            {"sentence": "朱元璋，即明[MASK]祖。", "expected": ["太", "高", "世"], "category": "皇帝称号"},
            
            # Military terms
            {"sentence": "从太祖渡[MASK]，积功由百夫长授元帅。", "expected": ["江", "河", "淮"], "category": "军事地理"},
            {"sentence": "帝念功臣[MASK]苦，特增其禄。", "expected": ["劳", "征", "战"], "category": "军功概念"},
            {"sentence": "兵部设[MASK]政司。", "expected": ["承", "都", "布"], "category": "军事机构"},
            {"sentence": "五军都督[MASK]。", "expected": ["府", "司", "院"], "category": "军事体制"},
            {"sentence": "京师设[MASK]营。", "expected": ["三", "五", "禁"], "category": "军事编制"},
            
            # Administrative terms
            {"sentence": "擢天策卫指挥同[MASK]。", "expected": ["知", "佥", "事"], "category": "官职"},
            {"sentence": "旋师取山[MASK]。", "expected": ["东", "西", "南"], "category": "地理方位"},
            {"sentence": "设[MASK]部尚书。", "expected": ["六", "各", "诸"], "category": "政府架构"},
            {"sentence": "都察院设左右[MASK]都御史。", "expected": ["副", "佥", "正"], "category": "监察体系"},
            {"sentence": "翰林院掌[MASK]诏。", "expected": ["制", "草", "撰"], "category": "文官职能"},
            
            # Classical patterns and literature
            {"sentence": "然非诸党人附[MASK]之。", "expected": ["丽", "从", "和"], "category": "古文语法"},
            {"sentence": "迨神宗末[MASK]，讹言朋兴。", "expected": ["年", "世", "时"], "category": "时间概念"},
            {"sentence": "衣冠填于[MASK]犴。", "expected": ["狴", "牢", "狱"], "category": "法律概念"},
            {"sentence": "士大夫以[MASK]为贵。", "expected": ["仕", "学", "德"], "category": "社会观念"},
            {"sentence": "庶民当[MASK]其业。", "expected": ["安", "勤", "专"], "category": "社会分工"},
            
            # Historical events and periods
            {"sentence": "成祖[MASK]北平，起兵靖难。", "expected": ["在", "居", "据"], "category": "历史事件"},
            {"sentence": "洪武[MASK]年，定天下。", "expected": ["元", "二", "三"], "category": "年号"},
            {"sentence": "建文帝[MASK]位四年。", "expected": ["在", "即", "登"], "category": "皇帝纪年"},
            {"sentence": "永乐迁都[MASK]京。", "expected": ["北", "应", "燕"], "category": "重大决策"},
            {"sentence": "嘉靖年间[MASK]寇猖獗。", "expected": ["倭", "海", "沿"], "category": "外患入侵"},
            
            # Government and administration
            {"sentence": "设六[MASK]，分理庶政。", "expected": ["部", "司", "曹"], "category": "政治制度"},
            {"sentence": "科举取[MASK]，三年一试。", "expected": ["士", "人", "才"], "category": "教育制度"},
            {"sentence": "内阁设[MASK]学士。", "expected": ["大", "翰", "侍"], "category": "内阁体制"},
            {"sentence": "御史巡[MASK]天下。", "expected": ["按", "视", "察"], "category": "监察制度"},
            {"sentence": "府县[MASK]民事。", "expected": ["理", "治", "管"], "category": "地方管理"},
            
            # Foreign relations and tribute
            {"sentence": "倭寇侵[MASK]，民不聊生。", "expected": ["边", "境", "海"], "category": "外患"},
            {"sentence": "诸蕃来[MASK]，皆有定制。", "expected": ["朝", "贡", "朝"], "category": "朝贡体系"},
            {"sentence": "西洋诸国遣使[MASK]贡。", "expected": ["朝", "进", "入"], "category": "对外关系"},
            {"sentence": "安南、暹罗皆为[MASK]国。", "expected": ["属", "藩", "朝"], "category": "册封体系"},
            
            # Economy and society
            {"sentence": "农民以[MASK]为本。", "expected": ["农", "田", "耕"], "category": "经济基础"},
            {"sentence": "商贾往来[MASK]市。", "expected": ["城", "各", "集"], "category": "商业活动"},
            {"sentence": "工匠各有[MASK]艺。", "expected": ["专", "技", "手"], "category": "手工业"},
            {"sentence": "赋税征收有[MASK]制。", "expected": ["定", "常", "法"], "category": "财政制度"},
            
            # Culture and education  
            {"sentence": "太学培养[MASK]才。", "expected": ["人", "英", "贤"], "category": "教育培养"},
            {"sentence": "儒学为[MASK]教。", "expected": ["正", "国", "王"], "category": "思想主导"},
            {"sentence": "诗文以[MASK]为宗。", "expected": ["古", "唐", "汉"], "category": "文学传统"},
            {"sentence": "医药典籍[MASK]传。", "expected": ["流", "世", "相"], "category": "医学发展"},
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
        
        # Expanded test sentences for comprehensive comparison
        test_sentences = [
            "太祖起[MASK]州，所至必克。",
            "明代[MASK]宦之祸酷矣。",
            "从太祖渡[MASK]，积功由百夫长授元帅。",
            "帝念功臣[MASK]苦，特增其禄。",
            "成祖[MASK]北平，起兵靖难。",
            "洪武[MASK]年，定天下。",
            "设六[MASK]，分理庶政。",
            "科举取[MASK]，三年一试。",
            "倭寇侵[MASK]，民不聊生。",
            "内阁设[MASK]学士。",
            "御史巡[MASK]天下。",
            "诸蕃来[MASK]，皆有定制。",
            "农民以[MASK]为本。",
            "儒学为[MASK]教。",
            "建文帝[MASK]位四年。"
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
    
    def create_visualization(self, output_dir: str = None, notebook_mode: bool = False):
        """Create evaluation visualization plots with enhanced Chinese font support
        
        Args:
            output_dir: Directory to save plots (if None and not notebook_mode, saves to current dir)
            notebook_mode: If True, displays plots inline instead of saving
        """
        
        # Set up Chinese font for better display
        import matplotlib.font_manager as fm
        
        # Configure matplotlib for Chinese characters - Enhanced approach
        try:
            # Try multiple Chinese font configuration strategies
            chinese_fonts = [
                'SimHei', 'Microsoft YaHei', 'Microsoft YaHei UI', 'PingFang SC', 
                'Hiragino Sans GB', 'STHeiti', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
                'Source Han Sans SC', 'Adobe Heiti Std', 'STSong', 'SimSun'
            ]
            
            font_found = False
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            # Method 1: Try system fonts by name
            for font_name in chinese_fonts:
                if font_name in available_fonts:
                    plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                    plt.rcParams['font.family'] = 'sans-serif'
                    font_found = True
                    logger.info(f"Using Chinese font: {font_name}")
                    break
            
            # Method 2: Try to find any Chinese font file
            if not font_found:
                for font_file in fm.findSystemFonts():
                    try:
                        font_prop = fm.FontProperties(fname=font_file)
                        font_name = font_prop.get_name()
                        # Check for Chinese characteristics
                        if (any(ord(char) > 127 for char in font_name) or 
                            any(name.lower() in font_name.lower() for name in 
                                ['simhei', 'yahei', 'pingfang', 'hiragino', 'stheiti', 'wenquanyi', 'noto', 'source han'])):
                            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                            plt.rcParams['font.family'] = 'sans-serif'
                            font_found = True
                            logger.info(f"Using Chinese font file: {font_name}")
                            break
                    except:
                        continue
            
            # Method 3: Platform-specific fallbacks
            if not font_found:
                import platform
                system = platform.system().lower()
                if 'windows' in system:
                    fallback_fonts = ['SimSun', 'Microsoft YaHei', 'SimHei']
                elif 'darwin' in system:  # macOS
                    fallback_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti']
                else:  # Linux
                    fallback_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'AR PL UMing CN']
                
                for font_name in fallback_fonts:
                    if font_name in available_fonts:
                        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                        font_found = True
                        logger.info(f"Using platform fallback font: {font_name}")
                        break
            
            if not font_found:
                logger.warning("No suitable Chinese fonts found - using default font")
                
        except Exception as e:
            logger.warning(f"Font configuration failed: {e} - using default font")
        
        # Enhanced matplotlib configuration for Chinese support
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.dpi'] = 100
        
        # Force font cache refresh if needed
        try:
            fm._load_fontmanager(try_read_cache=False)
        except:
            pass
        
        # Determine if we're in a notebook environment
        try:
            from IPython.display import display
            import matplotlib
            if notebook_mode or 'ipykernel' in str(type(get_ipython())):
                matplotlib.use('inline')
                notebook_mode = True
                logger.info("Notebook mode detected - displaying plots inline")
        except:
            notebook_mode = False
        
        if not notebook_mode and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Helper function to handle saving/displaying
        def save_or_show(fig, filename=None):
            if notebook_mode:
                plt.show()
            elif filename and output_dir:
                fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close(fig)
        
        # Test Chinese character rendering
        try:
            test_fig, test_ax = plt.subplots(figsize=(1, 1))
            test_ax.text(0.5, 0.5, '测试中文', ha='center', va='center', fontsize=12)
            test_ax.set_xlim(0, 1)
            test_ax.set_ylim(0, 1)
            test_ax.axis('off')
            if not notebook_mode:
                plt.close(test_fig)
            else:
                plt.close(test_fig)
            logger.info("Chinese character rendering test passed")
        except Exception as e:
            logger.warning(f"Chinese character test failed: {e}")
        
        # Always use bilingual titles for better compatibility
        use_bilingual = True
        
        # Plot 1: Accuracy by Category
        if 'prediction_accuracy' in self.results:
            acc_data = self.results['prediction_accuracy']['category_accuracies']
            if acc_data:
                fig = plt.figure(figsize=(14, 8))
                categories = list(acc_data.keys())
                accuracies = list(acc_data.values())
                
                # Create color gradient based on accuracy
                colors = ['#ff6b6b' if acc < 0.3 else '#ffa726' if acc < 0.6 else '#66bb6a' for acc in accuracies]
                
                bars = plt.bar(categories, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                if use_bilingual:
                    plt.title('Prediction Accuracy by Category | 领域特定预测准确率', 
                             fontsize=16, fontweight='bold', pad=20)
                    plt.xlabel('Category | 类别', fontsize=12)
                    plt.ylabel('Accuracy | 准确率', fontsize=12)
                else:
                    plt.title('Prediction Accuracy by Category', fontsize=16, fontweight='bold', pad=20)
                    plt.xlabel('Category', fontsize=12)
                    plt.ylabel('Accuracy', fontsize=12)
                
                plt.xticks(rotation=45, ha='right', fontsize=10)
                plt.ylim(0, 1.1)
                
                # Add value labels on bars
                for bar, acc in zip(bars, accuracies):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{acc:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
                
                # Add horizontal reference lines with bilingual labels
                plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, linewidth=1)
                plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, linewidth=1)
                plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=1)
                
                # Add performance zones
                plt.text(len(categories)-1, 0.85, 'Excellent | 优秀' if use_bilingual else 'Excellent', 
                        fontsize=9, color='green', fontweight='bold', ha='right')
                plt.text(len(categories)-1, 0.65, 'Good | 良好' if use_bilingual else 'Good', 
                        fontsize=9, color='orange', fontweight='bold', ha='right')
                plt.text(len(categories)-1, 0.35, 'Fair | 一般' if use_bilingual else 'Fair', 
                        fontsize=9, color='red', fontweight='bold', ha='right')
                
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                save_or_show(fig, 'accuracy_by_category.png')
        
        # Plot 2: Model Comparison - Side by Side Bars
        if 'baseline_comparison' in self.results:
            comp_data = self.results['baseline_comparison']['detailed_comparisons']
            if comp_data:
                fig = plt.figure(figsize=(16, 8))
                sentences = [f"Test {i+1}" for i in range(len(comp_data))]
                trained_scores = [d['trained_score'] for d in comp_data]
                baseline_scores = [d['baseline_score'] for d in comp_data]
                
                x = np.arange(len(sentences))
                width = 0.35
                
                # Create bars with different colors
                bars1 = plt.bar(x - width/2, trained_scores, width, 
                               label='Fine-tuned Model | 微调模型' if use_bilingual else 'Fine-tuned Model', 
                               color='#4CAF50', alpha=0.8, edgecolor='black', linewidth=0.5)
                bars2 = plt.bar(x + width/2, baseline_scores, width, 
                               label='Baseline BERT | 基线模型' if use_bilingual else 'Baseline BERT', 
                               color='#2196F3', alpha=0.8, edgecolor='black', linewidth=0.5)
                
                if use_bilingual:
                    plt.title('Model Performance Comparison | 模型性能对比', 
                             fontsize=16, fontweight='bold', pad=20)
                    plt.xlabel('Test Cases | 测试案例', fontsize=12)
                    plt.ylabel('Confidence Score | 置信度分数', fontsize=12)
                else:
                    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
                    plt.xlabel('Test Cases', fontsize=12)
                    plt.ylabel('Confidence Score', fontsize=12)
                
                plt.xticks(x, sentences, rotation=45, ha='right', fontsize=9)
                plt.legend(loc='upper right', fontsize=10)
                plt.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar, score in zip(bars1, trained_scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                for bar, score in zip(bars2, baseline_scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                plt.tight_layout()
                save_or_show(fig, 'model_comparison.png')
        
        # Plot 3: Improvement vs Baseline - Detailed Analysis
        if 'baseline_comparison' in self.results:
            comp_data = self.results['baseline_comparison']['detailed_comparisons']
            if comp_data:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
                
                # Left subplot: Improvement bar chart
                sentences = [f"Test {i+1}" for i in range(len(comp_data))]
                improvements = [d['score_improvement'] for d in comp_data]
                
                # Color code improvements: green for positive, red for negative
                colors = ['#4CAF50' if imp > 0 else '#f44336' for imp in improvements]
                
                bars = ax1.bar(sentences, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                if use_bilingual:
                    ax1.set_title('Score Improvement over Baseline | 相对基线的改进程度', fontweight='bold', fontsize=14)
                    ax1.set_xlabel('Test Cases | 测试案例', fontsize=12)
                    ax1.set_ylabel('Score Improvement | 分数改进', fontsize=12)
                else:
                    ax1.set_title('Score Improvement over Baseline', fontweight='bold', fontsize=14)
                    ax1.set_xlabel('Test Cases', fontsize=12)
                    ax1.set_ylabel('Score Improvement', fontsize=12)
                ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
                ax1.set_xticklabels(sentences, rotation=45, ha='right', fontsize=10)
                ax1.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bar, imp in zip(bars, improvements):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2, 
                            height + (0.01 if height >= 0 else -0.01),
                            f'{imp:+.3f}', ha='center', 
                            va='bottom' if height >= 0 else 'top',
                            fontweight='bold', fontsize=9)
                
                # Right subplot: Scatter plot comparison
                trained_scores = [d['trained_score'] for d in comp_data]
                baseline_scores = [d['baseline_score'] for d in comp_data]
                
                # Create scatter plot
                ax2.scatter(baseline_scores, trained_scores, c=colors, s=120, alpha=0.7, edgecolors='black', linewidth=1)
                
                # Add diagonal line (y=x) for reference
                min_score = min(min(trained_scores), min(baseline_scores))
                max_score = max(max(trained_scores), max(baseline_scores))
                ax2.plot([min_score, max_score], [min_score, max_score], 'k--', alpha=0.5, linewidth=2, 
                        label='Equal Performance | 相等性能' if use_bilingual else 'Equal Performance')
                
                if use_bilingual:
                    ax2.set_title('Fine-tuned vs Baseline Scores | 微调模型 vs 基线模型分数', fontweight='bold', fontsize=14)
                    ax2.set_xlabel('Baseline Score | 基线分数', fontsize=12)
                    ax2.set_ylabel('Fine-tuned Score | 微调分数', fontsize=12)
                else:
                    ax2.set_title('Fine-tuned vs Baseline Scores', fontweight='bold', fontsize=14)
                    ax2.set_xlabel('Baseline Score', fontsize=12)
                    ax2.set_ylabel('Fine-tuned Score', fontsize=12)
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
                
                # Add text annotations for each point
                for i, (base, trained) in enumerate(zip(baseline_scores, trained_scores)):
                    ax2.annotate(f'T{i+1}', (base, trained), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, alpha=0.8, fontweight='bold')
                
                plt.tight_layout()
                save_or_show(fig, 'improvement_analysis.png')
        
        # Plot 4: Overall Performance Summary Dashboard
        if 'prediction_accuracy' in self.results and 'baseline_comparison' in self.results:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Top-left: Overall accuracy gauge
            acc = self.results['prediction_accuracy']['overall_accuracy']
            sizes = [acc, 1-acc] if acc < 1 else [1]
            labels = [f'Accurate\n{acc:.1%}', f'Inaccurate\n{(1-acc):.1%}'] if acc < 1 else ['Perfect\n100%']
            colors_pie = ['#4caf50' if acc > 0.7 else '#ff9800' if acc > 0.4 else '#f44336', '#e0e0e0']
            
            wedges, texts, autotexts = ax1.pie(sizes, labels=labels[:len(sizes)], colors=colors_pie[:len(sizes)], 
                                              autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
            if use_bilingual:
                ax1.set_title('Overall Accuracy | 总体准确率', fontweight='bold', fontsize=14)
            else:
                ax1.set_title('Overall Accuracy', fontweight='bold', fontsize=14)
            
            # Top-right: Improvement distribution
            if 'baseline_comparison' in self.results:
                improvements = [d['score_improvement'] for d in self.results['baseline_comparison']['detailed_comparisons']]
                positive_improvements = [imp for imp in improvements if imp > 0]
                negative_improvements = [imp for imp in improvements if imp <= 0]
                
                improvement_data = [len(positive_improvements), len(negative_improvements)]
                if use_bilingual:
                    improvement_labels = [f'Improved | 改进\n{len(positive_improvements)} cases', 
                                        f'Same/Worse | 相同/更差\n{len(negative_improvements)} cases']
                else:
                    improvement_labels = [f'Improved\n{len(positive_improvements)} cases', 
                                        f'Same/Worse\n{len(negative_improvements)} cases']
                improvement_colors = ['#4caf50', '#f44336']
                
                ax2.pie(improvement_data, labels=improvement_labels, colors=improvement_colors, 
                       autopct='%1.0f', startangle=90, textprops={'fontsize': 9})
                if use_bilingual:
                    ax2.set_title('Improvement Distribution | 改进分布', fontweight='bold', fontsize=14)
                else:
                    ax2.set_title('Improvement Distribution', fontweight='bold', fontsize=14)
            
            # Bottom-left: Category performance horizontal bar chart
            if 'prediction_accuracy' in self.results:
                categories = list(self.results['prediction_accuracy']['category_accuracies'].keys())
                accuracies = list(self.results['prediction_accuracy']['category_accuracies'].values())
                
                y_pos = np.arange(len(categories))
                colors_cat = ['#4caf50' if acc > 0.7 else '#ff9800' if acc > 0.4 else '#f44336' for acc in accuracies]
                
                bars = ax3.barh(y_pos, accuracies, color=colors_cat, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(categories, fontsize=9)
                if use_bilingual:
                    ax3.set_xlabel('Accuracy | 准确率', fontsize=12)
                    ax3.set_title('Performance by Category | 各类别表现', fontweight='bold', fontsize=14)
                else:
                    ax3.set_xlabel('Accuracy', fontsize=12)
                    ax3.set_title('Performance by Category', fontweight='bold', fontsize=14)
                ax3.set_xlim(0, 1)
                ax3.grid(axis='x', alpha=0.3)
                
                # Add value labels
                for bar, acc in zip(bars, accuracies):
                    ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                            f'{acc:.2f}', ha='left', va='center', fontweight='bold', fontsize=9)
            
            # Bottom-right: Performance metrics summary
            ax4.axis('off')
            
            # Create summary statistics
            summary_text = []
            
            if 'prediction_accuracy' in self.results:
                acc = self.results['prediction_accuracy']['overall_accuracy']
                summary_text.append(f"Domain Accuracy: {acc:.1%}")
                
                # Performance rating
                if acc >= 0.8:
                    rating = "Excellent | 优秀" if use_bilingual else "Excellent"
                    rating_color = "#4caf50"
                elif acc >= 0.6:
                    rating = "Good | 良好" if use_bilingual else "Good"
                    rating_color = "#8bc34a"
                elif acc >= 0.4:
                    rating = "Fair | 一般" if use_bilingual else "Fair"
                    rating_color = "#ff9800"
                else:
                    rating = "Poor | 较差" if use_bilingual else "Poor"
                    rating_color = "#f44336"
                
                summary_text.append(f"Rating: {rating}")
            
            if 'baseline_comparison' in self.results:
                avg_imp = self.results['baseline_comparison']['average_score_improvement']
                summary_text.append(f"Avg Improvement: {avg_imp:+.3f}")
                
                improvements = [d['score_improvement'] for d in self.results['baseline_comparison']['detailed_comparisons']]
                positive_count = sum(1 for imp in improvements if imp > 0)
                total_count = len(improvements)
                summary_text.append(f"Cases Improved: {positive_count}/{total_count}")
            
            if 'perplexity' in self.results:
                perp = self.results['perplexity']
                summary_text.append(f"Perplexity: {perp:.1f}")
            
            # Display summary
            y_start = 0.9
            for i, text in enumerate(summary_text):
                ax4.text(0.1, y_start - i*0.12, text, fontsize=13, fontweight='bold',
                        transform=ax4.transAxes, verticalalignment='top')
            
            ax4.text(0.1, y_start - len(summary_text)*0.12 - 0.1, f"Overall Rating: {rating}", 
                    fontsize=15, fontweight='bold', color=rating_color,
                    transform=ax4.transAxes, verticalalignment='top')
            
            if use_bilingual:
                ax4.set_title('Performance Summary | 性能总结', fontweight='bold', fontsize=14, pad=20)
            else:
                ax4.set_title('Performance Summary', fontweight='bold', fontsize=14, pad=20)
            
            plt.tight_layout()
            save_or_show(fig, 'performance_summary.png')
        
        if not notebook_mode and output_dir:
            logger.info(f"Visualizations saved to {output_dir}")
            logger.info("Generated charts:")
            logger.info("  1. accuracy_by_category.png - Category-wise accuracy")
            logger.info("  2. model_comparison.png - Side-by-side model comparison")
            logger.info("  3. improvement_analysis.png - Detailed improvement analysis")
            logger.info("  4. performance_summary.png - Overall performance dashboard")
        else:
            logger.info("Charts displayed inline in notebook")

def main(notebook_mode: bool = False, model_path: str = None, data_folder: str = None, output_dir: str = None):
    """Main evaluation pipeline
    
    Args:
        notebook_mode: Whether running in notebook (displays plots inline)
        model_path: Path to the trained model
        data_folder: Path to test data folder  
        output_dir: Output directory for results
    """
    
    # Configuration - can be overridden by parameters
    MODEL_PATH = model_path or "chinese_ming_history_mlm"
    DATA_FOLDER = data_folder or "ming_history_chapters"
    OUTPUT_DIR = output_dir or "evaluation_results"
    BASELINE_MODEL = "bert-base-chinese"
    
    # Create output directory only if not in notebook mode
    if not notebook_mode and OUTPUT_DIR:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    logger.info("Starting comprehensive model evaluation...")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Data: {DATA_FOLDER}")
    logger.info(f"Baseline: {BASELINE_MODEL}")
    logger.info(f"Notebook mode: {notebook_mode}")
    
    try:
        # Load test data
        data_processor = MingHistoryDataProcessor(DATA_FOLDER)
        test_texts = data_processor.load_test_texts(max_files=100)
        
        if not test_texts:
            logger.error("No test texts loaded. Check data folder.")
            return None
        
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
        
        if not notebook_mode:
            # Save report
            report_path = Path(OUTPUT_DIR) / "evaluation_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Save detailed results
            evaluator.save_detailed_results(Path(OUTPUT_DIR) / "detailed_results.json")
            
            # Print summary
            print(f"\n📊 Evaluation Results Summary:")
            print(f"📝 Report: {report_path}")
            print(f"📊 Detailed Results: {Path(OUTPUT_DIR) / 'detailed_results.json'}")
            print(f"📈 Visualizations: {OUTPUT_DIR}/")
        
        # Create visualizations (inline if notebook mode)
        evaluator.create_visualization(OUTPUT_DIR if not notebook_mode else None, notebook_mode=notebook_mode)
        
        # Print report
        if not notebook_mode:
            print("\n" + report)
        
        logger.info("Evaluation completed successfully!")
        
        return evaluator  # Return evaluator for notebook use
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error("Stack trace:", exc_info=True)
        raise

def run_notebook_evaluation(model_path: str = "chinese_ming_history_mlm", 
                           data_folder: str = "ming_history_chapters"):
    """Convenience function for running evaluation in notebook
    
    Args:
        model_path: Path to the trained model
        data_folder: Path to test data folder
        
    Returns:
        ModelEvaluator: The evaluator with results
    """
    return main(notebook_mode=True, model_path=model_path, data_folder=data_folder)

if __name__ == "__main__":
    main()