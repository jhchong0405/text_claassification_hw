import numpy as np
from tensorflow.keras.datasets import imdb
from ml_classifier import TraditionalMLClassifier
from dl_classifier import DeepLearningClassifier
import torch
import tensorflow as tf

print("PyTorch version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

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

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Loading IMDB dataset...")
    x_train_text, x_test_text, y_train, y_test, x_train, x_test = load_imdb_data()
    
    # Traditional ML approach
    ml_classifier = TraditionalMLClassifier()
    ml_accuracy, ml_time = ml_classifier.train_and_evaluate(
        x_train_text, x_test_text, y_train, y_test
    )
    
    # Deep Learning approach
    dl_classifier = DeepLearningClassifier()
    dl_accuracy, dl_time = dl_classifier.train_and_evaluate(
        x_train, x_test, y_train, y_test
    )
    
    # Compare results
    print("\nComparison Summary:")
    print("-" * 50)
    print(f"Traditional ML - Accuracy: {ml_accuracy:.4f}, Time: {ml_time:.2f}s")
    print(f"Deep Learning - Accuracy: {dl_accuracy:.4f}, Time: {dl_time:.2f}s")

if __name__ == "__main__":
    main() 