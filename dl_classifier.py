import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import datetime

class DeepLearningClassifier:
    def __init__(self, model_name='roberta-base', max_length=128, batch_size=32, learning_rates=[2e-5]):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rates = learning_rates
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = SentimentClassifier(model_name).to(self.device)
        
    def _prepare_data(self, texts):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encodings
    
    def train_and_evaluate(self, x_train, x_test, y_train, y_test, log_file=None, train_size=5000, test_size=1000, save_checkpoints=False):
        def log(message):
            print(message)
            if log_file:
                log_file.write(message + '\n')
                log_file.flush()  # Ensure immediate writing
        
        log(f"\nDeep Learning Approach (Using {self.device}):")
        start_time = time.time()
        
        # Convert numerical sequences back to text
        word_index = tf.keras.datasets.imdb.get_word_index()
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        
        def decode_review(text):
            return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])
        
        # Convert to text and prepare small subset for quick testing
        x_train_text = [decode_review(x) for x in x_train[:train_size]]
        x_test_text = [decode_review(x) for x in x_test[:test_size]]
        y_train = y_train[:train_size]
        y_test = y_test[:test_size]
        
        # Prepare data
        log("Preparing data...")
        train_encodings = self._prepare_data(x_train_text)
        test_encodings = self._prepare_data(x_test_text)
        
        # Create dataloaders
        train_dataset = TensorDataset(
            train_encodings['input_ids'].to(self.device),
            train_encodings['attention_mask'].to(self.device),
            torch.tensor(y_train, dtype=torch.float32).to(self.device)
        )
        test_dataset = TensorDataset(
            test_encodings['input_ids'].to(self.device),
            test_encodings['attention_mask'].to(self.device),
            torch.tensor(y_test, dtype=torch.float32).to(self.device)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        # Training loop for different learning rates
        best_overall_accuracy = 0
        best_overall_report = None
        best_lr = None
        
        # Get today's date in YYYYMMDD format
        today_date = datetime.datetime.now().strftime("%Y%m%d")
        
        for lr in self.learning_rates:
            log(f"\nTesting with learning rate: {lr}")
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()
            
            best_accuracy = 0
            for epoch in range(2):  # Assuming 2 epochs for simplicity
                self.model.train()
                total_loss = 0
                
                log(f"\nEpoch {epoch + 1}/2")
                for batch in tqdm(train_loader, desc=f'Training'):
                    input_ids, attention_mask, labels = batch
                    
                    optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs.squeeze(), labels)
                    
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                # Evaluation
                accuracy, report, all_labels, all_preds = self.evaluate(test_loader)
                log(f'Epoch {epoch + 1}:')
                log(f'Average Loss: {total_loss/len(train_loader):.4f}')
                log(f'Test Accuracy: {accuracy:.4f}')
                log('\nClassification Report:')
                log(report)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_report = report
                    if save_checkpoints:
                        # Save best model for this learning rate
                        checkpoint_path = f'checkpoints/{today_date}_dl_model_lr{lr}_b{self.batch_size}_l{self.max_length}.pt'
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'accuracy': best_accuracy,
                            'classification_report': best_report
                        }, checkpoint_path)
                        log(f'Saved new best model with accuracy: {best_accuracy:.4f}')
            
            if best_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_accuracy
                best_overall_report = best_report
                best_lr = lr
        
        training_time = time.time() - start_time
        log(f"\nTraining completed:")
        log(f"Total Training Time: {training_time:.2f} seconds")
        log(f"Best Test Accuracy: {best_overall_accuracy:.4f} with learning rate: {best_lr}")
        
        # Calculate metrics
        from sklearn.metrics import recall_score, f1_score
        recall_neg = recall_score(all_labels, all_preds, pos_label=0)
        recall_pos = recall_score(all_labels, all_preds, pos_label=1)
        f1_neg = f1_score(all_labels, all_preds, pos_label=0)
        f1_pos = f1_score(all_labels, all_preds, pos_label=1)

        return best_overall_accuracy, training_time, best_overall_report, recall_neg, recall_pos, f1_neg, f1_pos
    
    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids, attention_mask)
                predicted = (outputs.squeeze() > 0).float()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        from sklearn.metrics import classification_report
        report = classification_report(all_labels, all_preds)
        accuracy = (all_preds == all_labels).mean()
        
        return accuracy, report, all_labels, all_preds

class SentimentClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.transformer.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        x = self.dropout(pooled_output)
        return self.fc(x)