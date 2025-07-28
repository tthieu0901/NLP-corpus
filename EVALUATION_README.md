# Model Evaluation Documentation

This document describes how to evaluate the Chinese Masked Language Model trained on Ming History corpus.

## Files Description

### Core Evaluation Scripts

1. **`eval_predict_word_model.py`** - Comprehensive evaluation script
   - Complete perplexity calculation on test data
   - Detailed accuracy analysis by category
   - Baseline model comparison
   - Semantic coherence evaluation
   - Vocabulary coverage analysis
   - Generates detailed reports and visualizations

2. **`quick_eval_model.py`** - Quick evaluation script
   - Fast domain-specific accuracy test
   - Basic baseline comparison
   - Coherence testing on sample texts
   - Generates summary report

3. **`run_evaluation.bat`** - Windows batch script for easy execution

### Evaluation Metrics

#### 1. Domain-Specific Accuracy
Tests the model's ability to predict masked words in historical Chinese contexts:
- **地名 (Place names)**: Geographic locations in historical context
- **历史概念 (Historical concepts)**: Dynasty-specific terms and concepts
- **政治制度 (Political systems)**: Government and administrative terms
- **军事概念 (Military concepts)**: Military titles and actions
- **教育制度 (Educational systems)**: Imperial examination terms
- **法律概念 (Legal concepts)**: Legal and judicial terminology

#### 2. Baseline Comparison
Compares the fine-tuned model with `bert-base-chinese`:
- Confidence score improvements
- Prediction accuracy differences
- Domain adaptation effectiveness

#### 3. Semantic Coherence
Evaluates contextual appropriateness of predictions:
- Character-level masked prediction
- Context-aware completions
- Classical Chinese pattern recognition

## Usage Instructions

### Prerequisites
```bash
pip install torch transformers datasets matplotlib seaborn scikit-learn
```

### Quick Evaluation (Recommended for first-time users)

#### Option 1: Using batch script (Windows)
```cmd
run_evaluation.bat
```

#### Option 2: Direct Python execution
```bash
python quick_eval_model.py
```

**Output files:**
- `quick_evaluation_report.txt` - Human-readable summary
- `quick_evaluation_results.json` - Detailed results in JSON format

### Comprehensive Evaluation

```bash
python eval_predict_word_model.py
```

**Output files:**
- `evaluation_results/evaluation_report.txt` - Detailed report
- `evaluation_results/detailed_results.json` - Complete results
- `evaluation_results/accuracy_by_category.png` - Accuracy visualization
- `evaluation_results/model_comparison.png` - Baseline comparison chart

## Understanding the Results

### Accuracy Scores
- **>= 70%**: Excellent domain adaptation
- **50-69%**: Good performance, suitable for most applications
- **30-49%**: Moderate performance, may need more training
- **< 30%**: Poor performance, requires training review

### Confidence Score Improvements
- **Positive values**: Model outperforms baseline
- **Negative values**: Baseline performs better (consider training adjustments)

### Perplexity (Comprehensive evaluation only)
- **< 50**: Excellent language modeling
- **50-100**: Good performance
- **> 100**: High uncertainty, needs improvement

## Sample Test Cases

### Historical Geography
```
Input:  太祖起[MASK]州，所至必克。
Expected: 濠, 凤, 淮
Context: Ming dynasty founder's place of origin
```

### Government Structure
```
Input:  设六[MASK]，分理庶政。
Expected: 部, 司, 曹
Context: Administrative departments in imperial government
```

### Imperial Examinations
```
Input:  科举取[MASK]，三年一试。
Expected: 士, 人, 才
Context: Imperial examination system for selecting officials
```

## Troubleshooting

### Common Issues

1. **Model not found error**
   ```
   Error: Trained model not found in 'chinese_ming_history_mlm' directory
   ```
   **Solution**: Run `build_predict_word_model.py` first to train the model.

2. **CUDA out of memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Use CPU-only mode by setting `device=-1` in pipeline initialization.

3. **Import errors**
   ```
   ModuleNotFoundError: No module named 'transformers'
   ```
   **Solution**: Install required packages:
   ```bash
   pip install torch transformers datasets matplotlib seaborn scikit-learn
   ```

### Performance Optimization

1. **For faster evaluation**: Use `quick_eval_model.py`
2. **For detailed analysis**: Use `eval_predict_word_model.py`
3. **Memory constraints**: Reduce batch size in comprehensive evaluation
4. **CPU-only systems**: Both scripts automatically detect and use CPU

## Interpreting Results

### Example Quick Evaluation Output
```
准确率: 7/10 = 70.0%
平均改进: +0.125
改进案例: 4/5

综合得分: 72.5%
🎉 优秀 - 模型训练成功，适合实际应用
```

This indicates:
- 70% accuracy on domain-specific tests
- Average confidence improvement of 0.125 over baseline
- Model outperforms baseline in 4 out of 5 test cases
- Overall score of 72.5% indicates excellent performance

### Recommendations Based on Results

#### Excellent Performance (70%+)
- ✅ Model ready for deployment
- ✅ Consider testing on larger datasets
- ✅ Suitable for production applications

#### Good Performance (50-69%)
- ✅ Model shows good domain adaptation
- ⚠️ Consider additional fine-tuning for higher accuracy
- ✅ Suitable for research and development

#### Needs Improvement (<50%)
- ❌ Review training data quality
- ❌ Adjust hyperparameters (learning rate, epochs)
- ❌ Consider using more training data

## Custom Evaluation

### Adding New Test Cases

Edit the test cases in `quick_eval_model.py`:

```python
test_cases = [
    {"text": "你的测试句子[MASK]这里。", "expected": ["期", "望", "答"], "type": "自定义类型"},
    # Add more test cases...
]
```

### Evaluating Different Model Checkpoints

```python
# Evaluate specific checkpoint
python quick_eval_model.py chinese_ming_history_mlm/checkpoint-epoch-20
```

## Best Practices

1. **Regular evaluation**: Run quick evaluation after every training session
2. **Comprehensive analysis**: Use full evaluation for final model assessment
3. **Baseline comparison**: Always compare with standard BERT models
4. **Domain relevance**: Focus on test cases relevant to your use case
5. **Performance tracking**: Keep evaluation logs for training progress monitoring

## Contact and Support

For questions about model evaluation:
1. Check the log files (`model_evaluation.log`, `mlm_training.log`)
2. Review the detailed results JSON files
3. Ensure all dependencies are properly installed
4. Verify model training completed successfully
