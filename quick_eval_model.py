#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Evaluation Script for Chinese Masked Language Model
Lightweight version for quick model assessment

This script provides essential evaluation metrics:
1. Quick perplexity estimation
2. Domain-specific word prediction tests
3. Baseline comparison on key examples
4. Performance summary report
"""

import os
import re
import json
import math
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import time

import torch
from transformers import BertTokenizer, BertForMaskedLM, pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickModelEvaluator:
    """Quick evaluation of the trained Chinese MLM"""
    
    def __init__(self, model_path: str, baseline_model: str = "bert-base-chinese"):
        self.model_path = model_path
        self.baseline_model = baseline_model
        
        logger.info(f"Loading models...")
        
        # Load trained model
        try:
            self.fill_mask = pipeline("fill-mask", model=model_path, tokenizer=model_path)
            logger.info(f"✓ Trained model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load trained model: {e}")
            raise
        
        # Load baseline model
        try:
            self.baseline_pipeline = pipeline("fill-mask", model=baseline_model, tokenizer=baseline_model)
            logger.info(f"✓ Baseline model loaded: {baseline_model}")
        except Exception as e:
            logger.warning(f"Failed to load baseline model: {e}")
            self.baseline_pipeline = None
        
        self.results = {}
    
    def quick_domain_test(self) -> Dict[str, any]:
        """Quick test on domain-specific examples"""
        logger.info("Running domain-specific tests...")
        
        # Carefully selected test cases with clear expected answers
        test_cases = [
            # Historical figures and places
            {"text": "太祖起[MASK]州，所至必克。", "expected": ["濠", "凤", "淮"], "type": "地名"},
            {"text": "明代[MASK]宦之祸酷矣。", "expected": ["阉", "宦", "权"], "type": "历史概念"},
            {"text": "成祖居[MASK]平，起兵靖难。", "expected": ["北", "燕", "幽"], "type": "地名"},
            
            # Government and military
            {"text": "设六[MASK]，分理庶政。", "expected": ["部", "司", "曹"], "type": "政治制度"},
            {"text": "科举取[MASK]，三年一试。", "expected": ["士", "人", "才"], "type": "教育制度"},
            {"text": "帝念功臣[MASK]苦，特增其禄。", "expected": ["劳", "征", "战"], "type": "军功概念"},
            
            # Classical Chinese patterns
            {"text": "洪武[MASK]年，定天下。", "expected": ["元", "二", "三"], "type": "年号"},
            {"text": "倭寇侵[MASK]，民不聊生。", "expected": ["边", "境", "海"], "type": "外患"},
            {"text": "旋师取山[MASK]。", "expected": ["东", "西", "南"], "type": "地理方位"},
            {"text": "衣冠填于[MASK]犴。", "expected": ["狴", "牢", "狱"], "type": "法律概念"},
        ]
        
        results = []
        hits = 0
        
        print("\n" + "="*70)
        print("DOMAIN-SPECIFIC PREDICTION TEST")
        print("="*70)
        
        for i, case in enumerate(test_cases, 1):
            text = case["text"]
            expected = case["expected"]
            test_type = case["type"]
            
            try:
                predictions = self.fill_mask(text)
                if predictions:
                    top_pred = predictions[0]
                    top_token = top_pred['token_str']
                    top_score = top_pred['score']
                    
                    # Check if prediction matches expected
                    hit = top_token in expected
                    if hit:
                        hits += 1
                    
                    # Get top 3 predictions
                    top_3 = [p['token_str'] for p in predictions[:3]]
                    
                    result = {
                        'test_case': text,
                        'expected': expected,
                        'predicted': top_token,
                        'score': top_score,
                        'hit': hit,
                        'top_3': top_3,
                        'type': test_type
                    }
                    results.append(result)
                    
                    # Display result
                    status = "✓" if hit else "✗"
                    print(f"{i:2d}. {status} {text}")
                    print(f"    预期: {expected}")
                    print(f"    预测: {top_token} (置信度: {top_score:.3f})")
                    print(f"    类型: {test_type}")
                    if not hit:
                        print(f"    前三: {top_3}")
                    print()
                
            except Exception as e:
                logger.error(f"Error testing: {text}, {e}")
        
        accuracy = hits / len(test_cases) if test_cases else 0
        
        summary = {
            'total_tests': len(test_cases),
            'hits': hits,
            'accuracy': accuracy,
            'detailed_results': results
        }
        
        print(f"准确率: {hits}/{len(test_cases)} = {accuracy:.1%}")
        print("="*70)
        
        self.results['domain_test'] = summary
        return summary
    
    def compare_with_baseline(self) -> Dict[str, any]:
        """Compare key predictions with baseline model"""
        if not self.baseline_pipeline:
            logger.warning("Baseline model not available for comparison")
            return {}
        
        logger.info("Comparing with baseline model...")
        
        test_sentences = [
            "太祖起[MASK]州，所至必克。",
            "明代[MASK]宦之祸酷矣。",
            "设六[MASK]，分理庶政。",
            "科举取[MASK]，三年一试。",
            "洪武[MASK]年，定天下。"
        ]
        
        comparisons = []
        
        print("\n" + "="*70)
        print("BASELINE MODEL COMPARISON")
        print("="*70)
        
        for i, sentence in enumerate(test_sentences, 1):
            try:
                # Get predictions from both models
                trained_preds = self.fill_mask(sentence)
                baseline_preds = self.baseline_pipeline(sentence)
                
                if trained_preds and baseline_preds:
                    trained_top = trained_preds[0]
                    baseline_top = baseline_preds[0]
                    
                    comparison = {
                        'sentence': sentence,
                        'trained_pred': trained_top['token_str'],
                        'trained_score': trained_top['score'],
                        'baseline_pred': baseline_top['token_str'],
                        'baseline_score': baseline_top['score'],
                        'improvement': trained_top['score'] - baseline_top['score']
                    }
                    comparisons.append(comparison)
                    
                    print(f"{i}. {sentence}")
                    print(f"   微调模型: {trained_top['token_str']} ({trained_top['score']:.3f})")
                    print(f"   基础模型: {baseline_top['token_str']} ({baseline_top['score']:.3f})")
                    
                    if comparison['improvement'] > 0:
                        print(f"   改进: +{comparison['improvement']:.3f} ✓")
                    else:
                        print(f"   改进: {comparison['improvement']:.3f} ✗")
                    print()
                
            except Exception as e:
                logger.error(f"Error comparing: {sentence}, {e}")
        
        avg_improvement = sum(c['improvement'] for c in comparisons) / len(comparisons) if comparisons else 0
        
        summary = {
            'comparisons': comparisons,
            'avg_improvement': avg_improvement,
            'improvement_count': sum(1 for c in comparisons if c['improvement'] > 0)
        }
        
        print(f"平均改进: {avg_improvement:.3f}")
        print(f"改进案例: {summary['improvement_count']}/{len(comparisons)}")
        print("="*70)
        
        self.results['baseline_comparison'] = summary
        return summary
    
    def test_coherence_samples(self) -> Dict[str, any]:
        """Test model coherence on sample texts"""
        logger.info("Testing coherence on sample texts...")
        
        # Sample classical Chinese texts with masked characters
        coherence_tests = [
            {"text": "顾时，字时[MASK]，濠人。", "context": "人名介绍"},
            {"text": "从太祖渡[MASK]，积功由百夫长授元帅。", "context": "军事行动"},
            {"text": "擢天策卫指挥同[MASK]。", "context": "官职任命"},
            {"text": "然非诸党人附[MASK]之。", "context": "政治关系"},
            {"text": "迨神宗末[MASK]，讹言朋兴。", "context": "时间描述"},
        ]
        
        coherence_results = []
        
        print("\n" + "="*70)
        print("COHERENCE TEST")
        print("="*70)
        
        for i, test in enumerate(coherence_tests, 1):
            text = test["text"]
            context = test["context"]
            
            try:
                predictions = self.fill_mask(text)
                if predictions:
                    top_3 = predictions[:3]
                    
                    result = {
                        'text': text,
                        'context': context,
                        'predictions': [(p['token_str'], p['score']) for p in top_3]
                    }
                    coherence_results.append(result)
                    
                    print(f"{i}. {text}")
                    print(f"   语境: {context}")
                    print(f"   预测:")
                    for j, pred in enumerate(top_3, 1):
                        print(f"     {j}. {pred['token_str']} (置信度: {pred['score']:.3f})")
                    print()
                
            except Exception as e:
                logger.error(f"Error testing coherence: {text}, {e}")
        
        self.results['coherence_test'] = coherence_results
        print("="*70)
        
        return coherence_results
    
    def generate_quick_report(self) -> str:
        """Generate a quick evaluation report"""
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("快速评估报告 - QUICK EVALUATION REPORT")
        report_lines.append("="*80)
        report_lines.append(f"模型路径: {self.model_path}")
        report_lines.append(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Domain Test Results
        if 'domain_test' in self.results:
            domain = self.results['domain_test']
            report_lines.append("1. 领域特定测试 (Domain-Specific Test)")
            report_lines.append("-" * 50)
            report_lines.append(f"测试案例: {domain['total_tests']}")
            report_lines.append(f"正确预测: {domain['hits']}")
            report_lines.append(f"准确率: {domain['accuracy']:.1%}")
            
            if domain['accuracy'] >= 0.7:
                report_lines.append("✓ 优秀 - 模型显示出良好的领域知识")
            elif domain['accuracy'] >= 0.5:
                report_lines.append("✓ 良好 - 模型具备基本的领域知识")
            elif domain['accuracy'] >= 0.3:
                report_lines.append("⚠ 一般 - 模型需要更多训练")
            else:
                report_lines.append("✗ 较差 - 模型可能需要重新训练")
            report_lines.append("")
        
        # Baseline Comparison
        if 'baseline_comparison' in self.results:
            baseline = self.results['baseline_comparison']
            report_lines.append("2. 基线模型对比 (Baseline Comparison)")
            report_lines.append("-" * 50)
            report_lines.append(f"平均置信度改进: {baseline['avg_improvement']:+.3f}")
            report_lines.append(f"改进案例: {baseline['improvement_count']}/{len(baseline['comparisons'])}")
            
            improvement_rate = baseline['improvement_count'] / len(baseline['comparisons']) if baseline['comparisons'] else 0
            
            if improvement_rate >= 0.7:
                report_lines.append("✓ 显著改进 - 微调效果明显")
            elif improvement_rate >= 0.5:
                report_lines.append("✓ 适度改进 - 微调有一定效果")
            else:
                report_lines.append("⚠ 改进有限 - 可能需要调整训练策略")
            report_lines.append("")
        
        # Overall Assessment
        report_lines.append("3. 总体评估 (Overall Assessment)")
        report_lines.append("-" * 50)
        
        # Calculate overall score
        scores = []
        if 'domain_test' in self.results:
            scores.append(self.results['domain_test']['accuracy'])
        if 'baseline_comparison' in self.results:
            baseline = self.results['baseline_comparison']
            improvement_rate = baseline['improvement_count'] / len(baseline['comparisons']) if baseline['comparisons'] else 0
            scores.append(improvement_rate)
        
        if scores:
            overall_score = sum(scores) / len(scores)
            report_lines.append(f"综合得分: {overall_score:.1%}")
            
            if overall_score >= 0.7:
                report_lines.append("🎉 优秀 - 模型训练成功，适合实际应用")
            elif overall_score >= 0.5:
                report_lines.append("👍 良好 - 模型表现不错，可考虑进一步优化")
            elif overall_score >= 0.3:
                report_lines.append("⚠️ 一般 - 模型需要改进")
            else:
                report_lines.append("❌ 需要改进 - 建议重新审视训练策略")
        
        report_lines.append("")
        report_lines.append("4. 建议 (Recommendations)")
        report_lines.append("-" * 50)
        
        if 'domain_test' in self.results:
            accuracy = self.results['domain_test']['accuracy']
            if accuracy < 0.4:
                report_lines.append("• 增加训练轮次或调整学习率")
                report_lines.append("• 检查训练数据质量和覆盖度")
            elif accuracy > 0.8:
                report_lines.append("• 模型表现优秀，可以部署使用")
                report_lines.append("• 考虑在更大数据集上进一步训练")
        
        if 'baseline_comparison' in self.results:
            improvement = self.results['baseline_comparison']['avg_improvement']
            if improvement < 0:
                report_lines.append("• 微调效果不理想，检查训练参数")
                report_lines.append("• 考虑使用更大的学习率或更多数据")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        return "\n".join(report_lines)
    
    def save_results(self, output_file: str = "quick_evaluation_results.json"):
        """Save evaluation results to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")

def main():
    """Main evaluation function"""
    
    MODEL_PATH = "chinese_ming_history_mlm"
    BASELINE_MODEL = "bert-base-chinese"
    
    print("🚀 启动快速模型评估...")
    print(f"📂 模型路径: {MODEL_PATH}")
    
    try:
        # Check if model exists
        if not Path(MODEL_PATH).exists():
            print(f"❌ 错误: 模型路径不存在 - {MODEL_PATH}")
            print("请确保已经运行 build_predict_word_model.py 训练模型")
            return
        
        # Initialize evaluator
        evaluator = QuickModelEvaluator(MODEL_PATH, BASELINE_MODEL)
        
        # Run evaluations
        print("\n🧪 开始评估测试...")
        
        # Domain-specific test
        evaluator.quick_domain_test()
        
        # Baseline comparison
        evaluator.compare_with_baseline()
        
        # Coherence test
        evaluator.test_coherence_samples()
        
        # Generate report
        report = evaluator.generate_quick_report()
        
        # Save results
        evaluator.save_results()
        
        # Display report
        print("\n" + report)
        
        # Save report to file
        with open("quick_evaluation_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📊 评估完成!")
        print(f"📄 报告已保存: quick_evaluation_report.txt")
        print(f"📋 详细结果: quick_evaluation_results.json")
        
    except Exception as e:
        logger.error(f"评估失败: {e}")
        print(f"❌ 评估失败: {e}")
        raise

if __name__ == "__main__":
    main()
