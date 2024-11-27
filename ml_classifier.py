import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import time

class TraditionalMLClassifier:
    def __init__(self, max_features=1000, min_df=5):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=0.7
        )
        
        # Initialize different classifiers with minimal parameters
        self.classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10),
            'Logistic Regression': LogisticRegression(max_iter=200),
            'Linear SVM': LinearSVC(max_iter=200),
            'Naive Bayes': MultinomialNB()
        }
        self.results = {}  # Store results for analysis
        
    def train_and_evaluate(self, x_train_text, x_test_text, y_train, y_test, log_file=None):
        def log(message):
            print(message)
            if log_file:
                log_file.write(message + '\n')
                log_file.flush()  # Ensure immediate writing
        
        log("\nTraditional ML Approaches:")
        start_time = time.time()
        
        # Use a subset of data for quick testing
        train_size = 5000
        test_size = 1000
        
        x_train_subset = x_train_text[:train_size]
        y_train_subset = y_train[:train_size]
        x_test_subset = x_test_text[:test_size]
        y_test_subset = y_test[:test_size]
        
        # Convert text to TF-IDF features
        log("Converting text to TF-IDF features...")
        x_train_tfidf = self.vectorizer.fit_transform(x_train_subset)
        x_test_tfidf = self.vectorizer.transform(x_test_subset)
        
        best_accuracy = 0
        best_model = None
        
        # Train and evaluate each classifier
        for name, classifier in self.classifiers.items():
            model_start_time = time.time()
            log(f"\nTraining {name}...")
            
            # Train
            classifier.fit(x_train_tfidf, y_train_subset)
            
            # Predict and evaluate
            y_pred = classifier.predict(x_test_tfidf)
            accuracy = accuracy_score(y_test_subset, y_pred)
            model_time = time.time() - model_start_time
            report = classification_report(y_test_subset, y_pred)
            
            self.results[name] = {
                'accuracy': accuracy,
                'time': model_time,
                'classification_report': report
            }
            
            log(f"\n{name} Results:")
            log(f"Training Time: {model_time:.2f} seconds")
            log(f"Accuracy: {accuracy:.4f}")
            log("\nClassification Report:")
            log(report)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = name
        
        total_time = time.time() - start_time
        
        log("\nOverall Results Summary:")
        log("-" * 50)
        for name, result in self.results.items():
            log(f"{name:20} - Accuracy: {result['accuracy']:.4f}, Time: {result['time']:.2f}s")
        log(f"\nBest Model: {best_model} (Accuracy: {best_accuracy:.4f})")
        log(f"Total Time: {total_time:.2f}s")
        
        return best_accuracy, total_time