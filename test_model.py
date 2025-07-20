#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the trained Chinese Masked Language Model
"""

import sys
from pathlib import Path
from transformers import pipeline, BertTokenizer, BertForMaskedLM

def test_trained_model(model_path="chinese_ming_history_mlm"):
    """Test the trained model with various masked sentences"""
    
    print("Loading trained model...")
    try:
        # Load the model using pipeline
        fill_mask = pipeline("fill-mask", model=model_path, tokenizer=model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test sentences with classical Chinese patterns
    test_sentences = [
        # Historical figures and titles
        "太祖起[MASK]州，所至必克。",  # Expected: 濠 (Hao)
        "明代[MASK]宦之祸酷矣。",      # Expected: 阉 (eunuch)
        "顾时，字时[MASK]，濠人。",    # Expected: 举 (ju)
        
        # Military and political terms
        "从太祖渡[MASK]，积功由百夫长授元帅。",  # Expected: 江 (river)
        "帝念功臣[MASK]苦，特增其禄。",        # Expected: 劳 (labor)
        
        # Administrative terms
        "擢天策卫指挥同[MASK]。",     # Expected: 知 (zhi)
        "旋师取山[MASK]。",           # Expected: 东 (dong)
        
        # Classical Chinese patterns
        "然非诸党人附[MASK]之。",      # Expected: 丽 (li)
        "迨神宗末[MASK]，讹言朋兴。",  # Expected: 年 (nian)
        "衣冠填于[MASK]犴。",         # Expected: 狴 (bi)
    ]
    
    print("\n" + "="*60)
    print("TESTING CHINESE MASKED LANGUAGE MODEL")
    print("="*60)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i}. Testing sentence: {sentence}")
        print("-" * 40)
        
        try:
            results = fill_mask(sentence)
            
            for j, result in enumerate(results[:3], 1):  # Top 3 predictions
                token = result['token_str']
                score = result['score']
                filled_sentence = result['sequence']
                print(f"  {j}. {token} (confidence: {score:.3f})")
                print(f"     Complete: {filled_sentence}")
                
        except Exception as e:
            print(f"     Error: {e}")
    
    print("\n" + "="*60)
    print("Testing completed!")

def interactive_test(model_path="chinese_ming_history_mlm"):
    """Interactive testing mode"""
    
    print("Loading model for interactive testing...")
    try:
        fill_mask = pipeline("fill-mask", model=model_path, tokenizer=model_path)
        print("Model loaded! Enter sentences with [MASK] token to test.")
        print("Type 'quit' to exit.\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    while True:
        try:
            sentence = input("Enter masked sentence: ").strip()
            
            if sentence.lower() in ['quit', 'exit', 'q']:
                break
                
            if '[MASK]' not in sentence:
                print("Please include [MASK] token in your sentence.")
                continue
            
            results = fill_mask(sentence)
            print("\nPredictions:")
            for i, result in enumerate(results[:5], 1):
                token = result['token_str']
                score = result['score']
                print(f"  {i}. {token} (confidence: {score:.3f})")
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("Goodbye!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "chinese_ming_history_mlm"
    
    print("Choose testing mode:")
    print("1. Automatic test with predefined sentences")
    print("2. Interactive testing")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        interactive_test(model_path)
    else:
        test_trained_model(model_path)
