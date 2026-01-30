#!/usr/bin/env python3
"""
Analyze overlap between models' forced choice final candidate pools.

Generates comprehensive statistics including:
- Pairwise overlap matrices
- Overall agreement measures
- Jaccard similarity coefficients
- Core consensus words
- Model-specific unique contributions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def load_model_candidates(file_path):
    """Load candidate words from a model's final pools CSV."""
    df = pd.read_csv(file_path)
    
    model_candidates = {}
    for _, row in df.iterrows():
        input_word = row['input']
        candidates_str = row['final_candidates']
        
        # Parse the comma-separated candidates string
        candidates = [word.strip().strip('"') for word in candidates_str.split(',')]
        candidates = [word for word in candidates if word]  # Remove empty strings
        
        model_candidates[input_word] = set(candidates)
    
    return model_candidates

def extract_model_name(filename):
    """Extract model name from filename."""
    return filename.split('_forced_choice_FINAL_pools_')[0]

def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def calculate_overlap_stats(model_data):
    """Calculate comprehensive overlap statistics between all model pairs."""
    models = list(model_data.keys())
    input_words = list(model_data[models[0]].keys())
    
    # Initialize results storage
    pairwise_jaccard = pd.DataFrame(index=models, columns=models, dtype=float)
    pairwise_intersection = pd.DataFrame(index=models, columns=models, dtype=int)
    pairwise_union = pd.DataFrame(index=models, columns=models, dtype=int)
    
    # Calculate pairwise statistics
    for model1, model2 in combinations(models, 2):
        total_jaccard = 0
        total_intersection = 0
        total_union = 0
        
        for input_word in input_words:
            set1 = model_data[model1][input_word]
            set2 = model_data[model2][input_word]
            
            jaccard = calculate_jaccard_similarity(set1, set2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            total_jaccard += jaccard
            total_intersection += intersection
            total_union += union
        
        # Average across all input words
        avg_jaccard = total_jaccard / len(input_words)
        avg_intersection = total_intersection / len(input_words)
        avg_union = total_union / len(input_words)
        
        # Fill symmetric matrix
        pairwise_jaccard.loc[model1, model2] = avg_jaccard
        pairwise_jaccard.loc[model2, model1] = avg_jaccard
        pairwise_intersection.loc[model1, model2] = avg_intersection
        pairwise_intersection.loc[model2, model1] = avg_intersection
        pairwise_union.loc[model1, model2] = avg_union
        pairwise_union.loc[model2, model1] = avg_union
    
    # Fill diagonal
    for model in models:
        pairwise_jaccard.loc[model, model] = 1.0
        # For self-comparison, intersection = union = pool size
        avg_pool_size = np.mean([len(model_data[model][word]) for word in input_words])
        pairwise_intersection.loc[model, model] = avg_pool_size
        pairwise_union.loc[model, model] = avg_pool_size
    
    return pairwise_jaccard, pairwise_intersection, pairwise_union

def find_consensus_words(model_data, min_models=None):
    """Find words that appear in candidate pools across multiple models."""
    models = list(model_data.keys())
    input_words = list(model_data[models[0]].keys())
    
    if min_models is None:
        min_models = len(models) // 2  # At least half the models
    
    consensus_results = {}
    
    for input_word in input_words:
        # Collect all candidates across models
        all_candidates = []
        for model in models:
            all_candidates.extend(model_data[model][input_word])
        
        # Count occurrences
        candidate_counts = Counter(all_candidates)
        
        # Find consensus candidates
        consensus_candidates = {
            candidate: count for candidate, count in candidate_counts.items()
            if count >= min_models
        }
        
        consensus_results[input_word] = {
            'total_unique_candidates': len(candidate_counts),
            'consensus_candidates': consensus_candidates,
            'consensus_count': len(consensus_candidates)
        }
    
    return consensus_results

def calculate_model_uniqueness(model_data):
    """Calculate how many unique candidates each model contributes."""
    models = list(model_data.keys())
    input_words = list(model_data[models[0]].keys())
    
    uniqueness_stats = {}
    
    for model in models:
        total_unique = 0
        total_candidates = 0
        
        for input_word in input_words:
            model_candidates = model_data[model][input_word]
            other_candidates = set()
            
            for other_model in models:
                if other_model != model:
                    other_candidates.update(model_data[other_model][input_word])
            
            unique_to_model = model_candidates - other_candidates
            total_unique += len(unique_to_model)
            total_candidates += len(model_candidates)
        
        uniqueness_stats[model] = {
            'total_candidates': total_candidates,
            'unique_candidates': total_unique,
            'uniqueness_rate': total_unique / total_candidates if total_candidates > 0 else 0
        }
    
    return uniqueness_stats

def create_visualizations(pairwise_jaccard, pairwise_intersection, output_dir):
    """Create heatmap visualizations of the overlap matrices."""
    
    # Jaccard similarity heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(pairwise_jaccard, dtype=bool), k=0)  # Mask diagonal and upper triangle
    sns.heatmap(pairwise_jaccard.astype(float), 
                annot=True, fmt='.3f', cmap='viridis',
                mask=mask, square=True, linewidths=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / 'jaccard_similarity_heatmap.pdf', bbox_inches='tight')
    plt.close()
    
    # Intersection size heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(pairwise_intersection.astype(float), 
                annot=True, fmt='.1f', cmap='plasma',
                mask=mask, square=True, linewidths=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / 'intersection_size_heatmap.pdf', bbox_inches='tight')
    plt.close()

def main():
    """Main analysis function."""
    # Define paths
    input_dir = Path("data/02_behavioral_associations/01_forced_choice/02_processed/final_candidates")
    output_dir = input_dir / "summary"
    output_dir.mkdir(exist_ok=True)
    
    # Find all model files
    model_files = list(input_dir.glob("*_forced_choice_FINAL_pools_*.csv"))
    
    if not model_files:
        print(f"No model files found in {input_dir}")
        return
    
    print(f"Found {len(model_files)} model files:")
    for f in model_files:
        print(f"  - {f.name}")
    
    # Load data from all models
    model_data = {}
    for file_path in model_files:
        model_name = extract_model_name(file_path.name)
        print(f"Loading data for {model_name}...")
        model_data[model_name] = load_model_candidates(file_path)
    
    models = list(model_data.keys())
    input_words = list(model_data[models[0]].keys())
    
    print(f"\nLoaded data for {len(models)} models and {len(input_words)} input words")
    
    # Calculate overlap statistics
    print("Calculating pairwise overlap statistics...")
    pairwise_jaccard, pairwise_intersection, pairwise_union = calculate_overlap_stats(model_data)
    
    # Find consensus words
    print("Finding consensus candidates...")
    consensus_results = find_consensus_words(model_data)
    
    # Calculate model uniqueness
    print("Calculating model uniqueness statistics...")
    uniqueness_stats = calculate_model_uniqueness(model_data)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    print("Saving results...")
    
    # 1. Pairwise matrices
    pairwise_jaccard.to_csv(output_dir / f"pairwise_jaccard_similarity_{timestamp}.csv")
    pairwise_intersection.to_csv(output_dir / f"pairwise_intersection_size_{timestamp}.csv")
    pairwise_union.to_csv(output_dir / f"pairwise_union_size_{timestamp}.csv")
    
    # 2. Overall statistics summary
    summary_stats = {
        'metric': [],
        'value': [],
        'description': []
    }
    
    # Average Jaccard similarity (excluding diagonal)
    jaccard_values = []
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            jaccard_values.append(pairwise_jaccard.iloc[i, j])
    
    avg_jaccard = np.mean(jaccard_values)
    std_jaccard = np.std(jaccard_values)
    min_jaccard = np.min(jaccard_values)
    max_jaccard = np.max(jaccard_values)
    
    summary_stats['metric'].extend(['avg_jaccard_similarity', 'std_jaccard_similarity', 
                                   'min_jaccard_similarity', 'max_jaccard_similarity'])
    summary_stats['value'].extend([avg_jaccard, std_jaccard, min_jaccard, max_jaccard])
    summary_stats['description'].extend([
        'Average Jaccard similarity across all model pairs',
        'Standard deviation of Jaccard similarities',
        'Minimum Jaccard similarity between any two models',
        'Maximum Jaccard similarity between any two models'
    ])
    
    # Consensus statistics
    total_consensus = sum(result['consensus_count'] for result in consensus_results.values())
    avg_consensus_per_word = total_consensus / len(input_words)
    
    summary_stats['metric'].extend(['total_consensus_candidates', 'avg_consensus_per_input'])
    summary_stats['value'].extend([total_consensus, avg_consensus_per_word])
    summary_stats['description'].extend([
        f'Total consensus candidates (appearing in ≥{len(models)//2} models)',
        'Average consensus candidates per input word'
    ])
    
    # Model uniqueness
    for model in models:
        summary_stats['metric'].append(f'{model}_uniqueness_rate')
        summary_stats['value'].append(uniqueness_stats[model]['uniqueness_rate'])
        summary_stats['description'].append(f'Proportion of unique candidates for {model}')
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / f"overlap_summary_statistics_{timestamp}.csv", index=False)
    
    # 3. Consensus candidates per input word
    consensus_df_data = []
    for input_word, result in consensus_results.items():
        for candidate, count in result['consensus_candidates'].items():
            consensus_df_data.append({
                'input_word': input_word,
                'candidate': candidate,
                'model_count': count,
                'proportion': count / len(models)
            })
    
    if consensus_df_data:
        try:
            consensus_df = pd.DataFrame(consensus_df_data)
            consensus_df = consensus_df.sort_values(['input_word', 'model_count'], ascending=[True, False])
            consensus_df.to_csv(output_dir / f"consensus_candidates_{timestamp}.csv", index=False)
        except OSError as e:
            print(f"Warning: Could not save consensus candidates CSV due to file system error: {e}")
            print("Continuing with other outputs...")
    
    # 4. Model uniqueness details
    uniqueness_df = pd.DataFrame(uniqueness_stats).T
    uniqueness_df.to_csv(output_dir / f"model_uniqueness_{timestamp}.csv")
    
    # 5. Create visualizations
    print("Creating visualizations...")
    create_visualizations(pairwise_jaccard, pairwise_intersection, output_dir)
    
    # Print summary to console
    print("\n" + "="*80)
    print("FORCED CHOICE MODEL OVERLAP ANALYSIS")
    print("="*80)
    
    print(f"\nModels analyzed: {len(models)}")
    for model in models:
        print(f"  - {model}")
    
    print(f"\nInput words: {len(input_words)}")
    print(f"Average pool size per model: {np.mean([len(model_data[model][word]) for model in models for word in input_words]):.1f}")
    
    print(f"\nOverall Agreement:")
    print(f"  - Average Jaccard similarity: {avg_jaccard:.3f} ± {std_jaccard:.3f}")
    print(f"  - Range: {min_jaccard:.3f} - {max_jaccard:.3f}")
    print(f"  - Total consensus candidates: {total_consensus}")
    print(f"  - Average consensus per input word: {avg_consensus_per_word:.1f}")
    
    print(f"\nModel Uniqueness:")
    for model in models:
        stats = uniqueness_stats[model]
        print(f"  - {model}: {stats['uniqueness_rate']:.1%} unique ({stats['unique_candidates']}/{stats['total_candidates']})")
    
    print(f"\nMost similar model pair:")
    max_idx = np.unravel_index(np.argmax(pairwise_jaccard.values + np.eye(len(models)) * -1), pairwise_jaccard.shape)
    model1, model2 = pairwise_jaccard.index[max_idx[0]], pairwise_jaccard.columns[max_idx[1]]
    print(f"  - {model1} ↔ {model2}: {pairwise_jaccard.loc[model1, model2]:.3f}")
    
    print(f"\nLeast similar model pair:")
    min_idx = np.unravel_index(np.argmin(pairwise_jaccard.values + np.eye(len(models)) * 2), pairwise_jaccard.shape)
    model1, model2 = pairwise_jaccard.index[min_idx[0]], pairwise_jaccard.columns[min_idx[1]]
    print(f"  - {model1} ↔ {model2}: {pairwise_jaccard.loc[model1, model2]:.3f}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Detailed matrices: *_{timestamp}.csv")
    print(f"  - Summary statistics: overlap_summary_statistics_{timestamp}.csv")
    print(f"  - Consensus candidates: consensus_candidates_{timestamp}.csv")
    print(f"  - Visualizations: *.pdf")

if __name__ == "__main__":
    main()