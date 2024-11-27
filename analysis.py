import matplotlib.pyplot as plt
import seaborn as sns
from ml_classifier import TraditionalMLClassifier
from dl_classifier import DeepLearningClassifier
import numpy as np
from tensorflow.keras.datasets import imdb
import time
import pandas as pd
import os
import json
import argparse


def load_imdb_data():
    # Load IMDB dataset
    max_features = 10000  # Maximum number of words to consider
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    
    # Convert indices back to words
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    # Decode reviews
    def decode_review(text):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])
    
    x_train_text = [decode_review(x) for x in x_train]
    x_test_text = [decode_review(x) for x in x_test]
    
    return x_train_text, x_test_text, y_train, y_test, x_train, x_test

def setup_directories():
    # Create necessary directories
    directories = ['plots', 'checkpoints', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_results_to_log(results_dict, filename):
    with open(os.path.join('logs', filename), 'w') as f:
        json.dump(results_dict, f, indent=4)

def experiment_ml_parameters(train_size=5000, test_size=1000):
    # Load data
    x_train_text, x_test_text, y_train, y_test, _, _ = load_imdb_data()
    
    # Parameters to test
    max_features_list = [500, 1000, 2000, 3000]
    min_df_list = [3, 5, 10]
    results = []
    
    # Setup logging
    log_file = open(os.path.join('logs', 'training_log.txt'), 'a')
    
    for max_features in max_features_list:
        for min_df in min_df_list:
            log_message = f"\nTesting ML with max_features={max_features}, min_df={min_df}"
            print(log_message)
            log_file.write(log_message + '\n')
            
            classifier = TraditionalMLClassifier(
                max_features=max_features,
                min_df=min_df
            )
            accuracy, train_time = classifier.train_and_evaluate(
                x_train_text, x_test_text, y_train, y_test,
                log_file=log_file,
                train_size=train_size,
                test_size=test_size
            )
            
            # Store results
            if hasattr(classifier, 'results'):
                for model_name, result in classifier.results.items():
                    results.append({
                        'max_features': max_features,
                        'min_df': min_df,
                        'model_name': model_name,
                        'accuracy': result['accuracy'],
                        'time': result['time'],
                        'classification_report': result.get('classification_report', '')
                    })
    
    log_file.close()
    save_results_to_log({'ml_results': results}, 'ml_results.json')
    return results

def experiment_dl_parameters(train_size=5000, test_size=1000):
    # Load data
    _, _, y_train, y_test, x_train, x_test = load_imdb_data()
    
    # Parameters to test
    batch_sizes = [16, 32, 64]
    max_lengths = [64, 128, 256]
    learning_rates = [1e-5, 2e-5, 3e-5]  # Add different learning rates
    results = []
    
    # Setup logging
    log_file = open(os.path.join('logs', 'training_log.txt'), 'a')
    
    for batch_size in batch_sizes:
        for max_length in max_lengths:
            for lr in learning_rates:
                log_message = f"\nTesting DL with batch_size={batch_size}, max_length={max_length}, learning_rate={lr}"
                print(log_message)
                log_file.write(log_message + '\n')
                
                classifier = DeepLearningClassifier(
                    max_length=max_length,
                    batch_size=batch_size,
                    learning_rates=[lr]  # Pass the current learning rate
                )
                accuracy, train_time, report, recall_neg, recall_pos, f1_neg, f1_pos = classifier.train_and_evaluate(
                    x_train, x_test, y_train, y_test,
                    log_file=log_file,
                    train_size=train_size,
                    test_size=test_size
                )
                
                results.append({
                    'batch_size': batch_size,
                    'max_length': max_length,
                    'learning_rate': lr,
                    'accuracy': accuracy,
                    'time': train_time,
                    'classification_report': report,
                    'model_name': 'RoBERTa',
                    'recall_neg': recall_neg,
                    'recall_pos': recall_pos,
                    'f1_neg': f1_neg,
                    'f1_pos': f1_pos
                })
    
    log_file.close()
    save_results_to_log({'dl_results': results}, 'dl_results.json')
    return results

def save_plots(ml_results, dl_results):
    ml_df = pd.DataFrame(ml_results)
    dl_df = pd.DataFrame(dl_results)
    
    # 1. ML Models Comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='model_name', y='accuracy', data=ml_df)
    plt.title('Accuracy Distribution by ML Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/ml_models_comparison.png')
    plt.close()
    
    # 2. ML Features Impact
    plt.figure(figsize=(12, 6))
    for model in ml_df['model_name'].unique():
        model_data = ml_df[ml_df['model_name'] == model]
        plt.plot(model_data['max_features'], model_data['accuracy'], 
                marker='o', label=model)
    plt.title('Impact of Max Features on ML Models')
    plt.xlabel('Max Features')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/ml_features_impact.png')
    plt.close()
    
    # 3. DL Parameters Impact
    plt.figure(figsize=(10, 6))
    pivot_table = dl_df.pivot_table(
        index='batch_size', 
        columns='max_length', 
        values='accuracy'
    )
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('DL Accuracy by Batch Size and Max Length')
    plt.tight_layout()
    plt.savefig('plots/dl_parameters_impact.png')
    plt.close()
    
    # 4. Training Time Comparison
    plt.figure(figsize=(10, 6))
    time_comparison = {
        'ML (avg)': ml_df['time'].mean(),
        'DL (avg)': dl_df['time'].mean()
    }
    plt.bar(time_comparison.keys(), time_comparison.values(), color=['blue', 'orange'])
    plt.title('Average Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('plots/training_time_comparison.png')
    plt.close()
    
    # 5. ML Models Time vs Accuracy
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=ml_df, x='time', y='accuracy', hue='model_name', style='model_name', s=100)
    plt.title('ML Models: Training Time vs Accuracy')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/ml_time_vs_accuracy.png')
    plt.close()
    
    # 6. DL Learning Rate Impact
    plt.figure(figsize=(12, 6))
    for batch_size in dl_df['batch_size'].unique():
        for max_length in dl_df['max_length'].unique():
            subset = dl_df[(dl_df['batch_size'] == batch_size) & (dl_df['max_length'] == max_length)]
            plt.plot(subset['learning_rate'], subset['accuracy'], marker='o', label=f'Batch {batch_size}, Length {max_length}')
    
    plt.title('Impact of Learning Rate on DL Model Accuracy')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.xscale('log')  # Log scale for learning rates
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/dl_learning_rate_impact.png')
    plt.close()
    
    # Save results summary
    with open('plots/results_summary.txt', 'w') as f:
        f.write("Best Results Summary:\n")
        f.write("-" * 50 + "\n")
        
        best_ml = ml_df.loc[ml_df['accuracy'].idxmax()]
        f.write(f"Best ML Model: {best_ml['model_name']}\n")
        f.write(f"Parameters: max_features={best_ml['max_features']}, min_df={best_ml['min_df']}\n")
        f.write(f"Accuracy: {best_ml['accuracy']:.4f}\n")
        f.write(f"Training Time: {best_ml['time']:.2f}s\n\n")
        
        best_dl = dl_df.loc[dl_df['accuracy'].idxmax()]
        f.write(f"Best DL Model:\n")
        f.write(f"Parameters: batch_size={best_dl['batch_size']}, max_length={best_dl['max_length']}\n")
        f.write(f"Accuracy: {best_dl['accuracy']:.4f}\n")
        f.write(f"Training Time: {best_dl['time']:.2f}s\n")

def load_and_show_results():
    # Load results from JSON files
    try:
        with open('logs/ml_results.json', 'r') as f:
            ml_results = json.load(f)['ml_results']
        with open('logs/dl_results.json', 'r') as f:
            dl_results = json.load(f)['dl_results']
            
        # Display results
        print("\nMachine Learning Results:")
        print("-" * 50)
        ml_df = pd.DataFrame(ml_results)
        print(ml_df.groupby('model_name')['accuracy'].agg(['mean', 'max', 'min']))
        
        print("\nDeep Learning Results:")
        print("-" * 50)
        dl_df = pd.DataFrame(dl_results)
        print(dl_df.groupby(['batch_size', 'max_length'])['accuracy'].mean().unstack())
        
        # Generate plots
        save_plots(ml_results, dl_results)
        print("\nPlots have been updated in the 'plots' directory")
        
    except FileNotFoundError:
        print("No results found. Please run training first.")

def create_comparative_plots(ml_results, dl_results):
    ml_df = pd.DataFrame(ml_results)
    dl_df = pd.DataFrame(dl_results)
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots/comparison', exist_ok=True)
    
    # 1. Overall Accuracy Comparison
    plt.figure(figsize=(12, 6))
    
    # ML Models
    ml_accuracies = ml_df.groupby('model_name')['accuracy'].max()
    plt.bar(range(len(ml_accuracies)), ml_accuracies.values, 
            label='ML Models', alpha=0.8, color='skyblue')
    
    # Best DL Model
    best_dl_acc = dl_df['accuracy'].max()
    plt.bar(len(ml_accuracies), best_dl_acc, 
            label='Best DL Model', alpha=0.8, color='orange')
    
    plt.xticks(range(len(ml_accuracies) + 1), 
               list(ml_accuracies.index) + ['RoBERTa'], 
               rotation=45)
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison: ML vs DL Models')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('plots/comparison/accuracy_comparison.png')
    plt.close()
    
    # 2. Training Time vs Accuracy Scatter Plot
    plt.figure(figsize=(12, 6))
    
    # ML Models
    for model in ml_df['model_name'].unique():
        model_data = ml_df[ml_df['model_name'] == model]
        plt.scatter(model_data['time'], model_data['accuracy'], 
                   label=model, alpha=0.6)
    
    # DL Models
    plt.scatter(dl_df['time'], dl_df['accuracy'], 
                label='RoBERTa', alpha=0.6, marker='*', s=200)
    
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title('Training Time vs Accuracy: ML vs DL Models')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/comparison/time_vs_accuracy.png')
    plt.close()
    
    # 3. Box Plot Comparison
    plt.figure(figsize=(10, 6))
    
    # Prepare data for boxplot
    all_accuracies = []
    labels = []
    
    for model in ml_df['model_name'].unique():
        all_accuracies.append(ml_df[ml_df['model_name'] == model]['accuracy'])
        labels.append(model)
    
    all_accuracies.append(dl_df['accuracy'])
    labels.append('RoBERTa')
    
    plt.boxplot(all_accuracies, labels=labels)
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    plt.title('Accuracy Distribution: ML vs DL Models')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('plots/comparison/accuracy_distribution.png')
    plt.close()
    
    # Save comparative summary
    with open('plots/comparison/comparative_summary.txt', 'w') as f:
        f.write("Comparative Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # ML Models Summary
        f.write("Machine Learning Models:\n")
        f.write("-" * 30 + "\n")
        for model in ml_df['model_name'].unique():
            model_data = ml_df[ml_df['model_name'] == model]
            f.write(f"{model}:\n")
            f.write(f"  Best Accuracy: {model_data['accuracy'].max():.4f}\n")
            f.write(f"  Average Accuracy: {model_data['accuracy'].mean():.4f}\n")
            f.write(f"  Average Training Time: {model_data['time'].mean():.2f}s\n\n")
        
        # DL Model Summary
        f.write("\nDeep Learning Model (RoBERTa):\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best Accuracy: {dl_df['accuracy'].max():.4f}\n")
        f.write(f"Average Accuracy: {dl_df['accuracy'].mean():.4f}\n")
        f.write(f"Average Training Time: {dl_df['time'].mean():.2f}s\n\n")
        
        # Overall Comparison
        f.write("\nOverall Comparison:\n")
        f.write("-" * 30 + "\n")
        best_ml_acc = ml_df['accuracy'].max()
        best_dl_acc = dl_df['accuracy'].max()
        f.write(f"Best ML Accuracy: {best_ml_acc:.4f}\n")
        f.write(f"Best DL Accuracy: {best_dl_acc:.4f}\n")
        f.write(f"Accuracy Difference: {(best_dl_acc - best_ml_acc):.4f}\n")

def extract_metrics_from_report(report):
    # Assuming the report is a string in the format of sklearn's classification_report
    lines = report.split('\n')
    metrics = {
        'recall_neg': 0.0,
        'recall_pos': 0.0,
        'f1_neg': 0.0,
        'f1_pos': 0.0
    }
    
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue  # Skip lines that don't have enough parts
        if parts[0] == '0':  # Assuming '0' is the label for the negative class
            metrics['recall_neg'] = float(parts[2])  # Recall is the third value
            metrics['f1_neg'] = float(parts[3])      # F1-score is the fourth value
        elif parts[0] == '1':  # Assuming '1' is the label for the positive class
            metrics['recall_pos'] = float(parts[2])  # Recall is the third value
            metrics['f1_pos'] = float(parts[3])      # F1-score is the fourth value
    
    return metrics

def create_metric_comparison_plots(ml_results, dl_results):
    # Create directory for metric comparisons
    os.makedirs('plots/metrics', exist_ok=True)
    
    ml_df = pd.DataFrame(ml_results)
    dl_df = pd.DataFrame(dl_results)
    
    # Collect metrics for ML models
    metrics_data = {
        'model_names': [],
        'accuracies': [],
        'recalls_neg': [],
        'recalls_pos': [],
        'f1_neg': [],
        'f1_pos': []
    }
    
    # Process ML models
    for model in ml_df['model_name'].unique():
        if model != 'RoBERTa':  # Skip DL model in ML results
            model_data = ml_df[ml_df['model_name'] == model].iloc[0]
            metrics_data['model_names'].append(model)
            metrics_data['accuracies'].append(model_data['accuracy'])
            if 'classification_report' in model_data:
                metrics = extract_metrics_from_report(model_data['classification_report'])
                metrics_data['recalls_neg'].append(metrics['recall_neg'])
                metrics_data['recalls_pos'].append(metrics['recall_pos'])
                metrics_data['f1_neg'].append(metrics['f1_neg'])
                metrics_data['f1_pos'].append(metrics['f1_pos'])
    
    # Add DL model metrics
    best_dl = dl_df.loc[dl_df['accuracy'].idxmax()]
    metrics_data['model_names'].append('RoBERTa')
    metrics_data['accuracies'].append(best_dl['accuracy'])
    metrics_data['recalls_neg'].append(float(best_dl['recall_neg']))
    metrics_data['recalls_pos'].append(float(best_dl['recall_pos']))
    metrics_data['f1_neg'].append(float(best_dl['f1_neg']))
    metrics_data['f1_pos'].append(float(best_dl['f1_pos']))
    
    if not metrics_data['model_names']:  # If no metrics were collected
        print("No valid metrics found in the results")
        return
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics_data['model_names']))
    plt.bar(x, metrics_data['accuracies'], color='skyblue')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison Across Models')
    plt.xticks(x, metrics_data['model_names'], rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('plots/metrics/accuracy_comparison.png')
    plt.close()
    
    # 2. Recall Comparison
    plt.figure(figsize=(12, 6))
    width = 0.35
    
    plt.bar(x - width/2, metrics_data['recalls_neg'], width, 
            label='Negative Class', color='lightcoral')
    plt.bar(x + width/2, metrics_data['recalls_pos'], width, 
            label='Positive Class', color='lightblue')
    
    plt.xlabel('Models')
    plt.ylabel('Recall')
    plt.title('Recall Comparison Across Models')
    plt.xticks(x, metrics_data['model_names'], rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('plots/metrics/recall_comparison.png')
    plt.close()
    
    # 3. F1-Score Comparison
    plt.figure(figsize=(12, 6))
    
    plt.bar(x - width/2, metrics_data['f1_neg'], width, 
            label='Negative Class', color='lightcoral')
    plt.bar(x + width/2, metrics_data['f1_pos'], width, 
            label='Positive Class', color='lightblue')
    
    plt.xlabel('Models')
    plt.ylabel('F1-Score')
    plt.title('F1-Score Comparison Across Models')
    plt.xticks(x, metrics_data['model_names'], rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('plots/metrics/f1_comparison.png')
    plt.close()
    
    # 4. Recall Index Plot
    plt.figure(figsize=(12, 6))
    recall_index = [(pos - neg) for pos, neg in zip(metrics_data['recalls_pos'], metrics_data['recalls_neg'])]
    plt.bar(x, recall_index, color='purple')
    
    plt.xlabel('Models')
    plt.ylabel('Recall Index (Positive - Negative)')
    plt.title('Recall Index Across Models')
    plt.xticks(x, metrics_data['model_names'], rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('plots/metrics/recall_index.png')
    plt.close()
    
    # 5. Metrics Summary Table
    metrics_summary = pd.DataFrame({
        'Model': metrics_data['model_names'],
        'Accuracy': metrics_data['accuracies'],
        'Negative Recall': metrics_data['recalls_neg'],
        'Positive Recall': metrics_data['recalls_pos'],
        'Negative F1': metrics_data['f1_neg'],
        'Positive F1': metrics_data['f1_pos'],
        'Avg Recall': [(n + p)/2 for n, p in zip(metrics_data['recalls_neg'], 
                                                metrics_data['recalls_pos'])],
        'Avg F1': [(n + p)/2 for n, p in zip(metrics_data['f1_neg'], 
                                            metrics_data['f1_pos'])]
    })
    # Save metrics summary
    with open('plots/metrics/metrics_summary.txt', 'w') as f:
        f.write("Detailed Metrics Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(metrics_summary.to_string())
        
        f.write("\n\nBest Models by Metric:\n")
        f.write("-" * 30 + "\n")
        
        metrics_to_check = ['Accuracy', 'Avg Recall', 'Avg F1']
        for metric in metrics_to_check:
            best_idx = metrics_summary[metric].idxmax()
            f.write(f"\nBest {metric}:\n")
            f.write(f"Model: {metrics_summary.loc[best_idx, 'Model']}\n")
            f.write(f"Value: {metrics_summary.loc[best_idx, metric]:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description='IMDB Classification Analysis')
    parser.add_argument('--show-only', action='store_true', 
                        help='Only show results without training')
    parser.add_argument('--train-size', type=int, default=5000, 
                        help='Size of the training dataset')
    parser.add_argument('--test-size', type=int, default=1000, 
                        help='Size of the testing dataset')
    args = parser.parse_args()
    
    setup_directories()
    
    if args.show_only:
        load_and_show_results()
    else:
        print("Running ML experiments...")
        ml_results = experiment_ml_parameters(train_size=args.train_size, test_size=args.test_size)
        
        print("\nRunning DL experiments...")
        dl_results = experiment_dl_parameters(train_size=args.train_size, test_size=args.test_size)
        
        print("\nGenerating and saving plots...")
        save_plots(ml_results, dl_results)
        create_comparative_plots(ml_results, dl_results)
        create_metric_comparison_plots(ml_results, dl_results)
        print("Results saved in 'plots' directory")

if __name__ == "__main__":
    main() 