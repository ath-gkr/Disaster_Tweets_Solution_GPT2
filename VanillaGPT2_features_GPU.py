import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, input_a, input_b, input_c, labels):
        self.input_a = input_a
        self.input_b = input_b
        self.input_c = input_c
        self.labels = labels
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoded_inputs_a = self.tokenizer.encode_plus(
            self.input_a[index],
            add_special_tokens=True,
            max_length=120,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        encoded_inputs_b = self.tokenizer.encode_plus(
            self.input_b[index],
            add_special_tokens=True,
            max_length=120,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_a = encoded_inputs_a['input_ids'].squeeze()
        input_b = encoded_inputs_b['input_ids'].squeeze()
        input_c = torch.tensor(self.input_c[index]).unsqueeze(0)
        labels = torch.tensor(self.labels[index]).unsqueeze(0)

        return input_a, input_b, input_c, labels



# Define the GPT2BinaryClassifier
class GPT2BinaryClassifier(nn.Module):
    def __init__(self, gpt2_model, num_classes):
        super(GPT2BinaryClassifier, self).__init__()

        # GPT2 Transformer
        self.gpt2_model = gpt2_model
               
        # Classification layers
        gpt2_output_size = self.gpt2_model.config.hidden_size
        input_c_size = 1  # Size of input_c
        linear_input_size = gpt2_output_size + input_c_size
        self.classifier = nn.Sequential(
            nn.Linear(linear_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # Binary classification, so using Sigmoid activation
        )

    def forward(self, input_a, input_b, input_c):
        input_concat = torch.cat((input_a, input_b), dim=1)  # Concatenate input_a and input_b horizontally
        gpt2_output = self.gpt2_model(input_concat).last_hidden_state[:, 0, :]
        input_c = input_c.unsqueeze(1)  # Add an extra dimension to input_c
        combined_input = torch.cat((gpt2_output, input_c), dim=1)
        logits = self.classifier(combined_input).squeeze().float()
        
        return logits

# Define a training function
def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0

    for input_batch in train_loader:
        
        for batch_idx in range(input_batch[2].size(0)):  #Because last batch smaller range(batch_size):
            input_a = input_batch[0][batch_idx].unsqueeze(0).to(device)
            input_b = input_batch[1][batch_idx].unsqueeze(0).to(device)
            input_c = input_batch[2][batch_idx].to(device) #Do NOT unsqueeze this!
            labels = input_batch[3][batch_idx].squeeze().to(device)   
            

            optimizer.zero_grad()

            output = model(input_a, input_b, input_c)
            labels = labels.float()  #Because it complains in the criterion...
           
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

    train_loss /= len(train_loader)
    return train_loss

# Define a validation function
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for input_batch in val_loader:
            #print('input_batch_a', input_batch[0].size(), 'input_batch_b', input_batch[1].size(),'input_batch_c', input_batch[2].size(),'input_batch_d', input_batch[3].size())
            for batch_idx in range(input_batch[2].size(0)):  #Because last batch smaller
                input_a = input_batch[0][batch_idx].unsqueeze(0).to(device)
                input_b = input_batch[1][batch_idx].unsqueeze(0).to(device)
                input_c = input_batch[2][batch_idx].to(device) #Do NOT unsqueeze this!
                labels = input_batch[3][batch_idx].squeeze().to(device)
       
                output = model(input_a, input_b, input_c)
                labels = labels.float()  # Reshape labels to match the output shape
                loss = criterion(output, labels)

                val_loss += loss.item()

                output = torch.sigmoid(output)  # Apply sigmoid activation
                predicted_labels = torch.round(output)  # Round predictions to binary values

                predictions.extend(predicted_labels.flatten().cpu().type(torch.int32).tolist())
                targets.extend(labels.flatten().cpu().type(torch.int32).tolist())

    val_loss /= len(val_loader)
    return val_loss, predictions, targets

# Load data from pandas DataFrame
train_df_clean = pd.read_csv("train_clean_processed.csv") 
input_a = train_df_clean["text_clean"].values
input_b = train_df_clean["hashtags"].values
input_c = train_df_clean["num_exclamations"].values
labels = train_df_clean["target"].values

# Convert labels to integers if necessary
labels = labels.astype(int)

# Set hyperparameters
gpt2_model_name = 'gpt2'
num_classes = 1
num_epochs = 12
batch_size = 50
learning_rate = 1e-4
num_folds = 2  # Number of folds for cross-validation

# Initialize evaluation metrics
avg_acc, avg_precision, avg_recall, avg_f1 = 0.0, 0.0, 0.0, 0.0

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=["Fold", "Epoch", "Train Loss", "Val Loss", "Accuracy", "Precision", "Recall", "F1-Score"])

# Perform cross-validation
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(input_a, labels)):
    print(f"Fold {fold+1}:")

    # Split data into training and validation sets
    train_a, train_b, train_c, train_labels = input_a[train_idx], input_b[train_idx], input_c[train_idx], labels[train_idx]
    val_a, val_b, val_c, val_labels = input_a[val_idx], input_b[val_idx], input_c[val_idx], labels[val_idx]

    # Create custom datasets and data loaders
    train_dataset = CustomDataset(train_a, train_b, train_c, train_labels)
    val_dataset = CustomDataset(val_a, val_b, val_c, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create the tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

    # Create model instance and define loss function and optimizer
    gpt2_model = GPT2Model.from_pretrained(gpt2_model_name).to(device)
    model = GPT2BinaryClassifier(gpt2_model, num_classes).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        print('Started training in epoch', epoch+1)
        train_loss = train(model, train_loader, criterion, optimizer)
        print('Started validation in epoch', epoch+1)
        val_loss, predictions, targets = validate(model, val_loader, criterion)

        acc = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions)
        recall = recall_score(targets, predictions)
        f1 = f1_score(targets, predictions)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        print(f"Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        # Append the results to the DataFrame
        results_df = results_df.append({
            "Fold": fold + 1,
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }, ignore_index=True)

    # Accumulate evaluation metrics
    avg_acc += acc
    avg_precision += precision
    avg_recall += recall
    avg_f1 += f1

# Compute average evaluation metrics
avg_acc /= num_folds
avg_precision /= num_folds
avg_recall /= num_folds
avg_f1 /= num_folds

# Print average evaluation metrics
print("Average Metrics:")
print(f"Accuracy={avg_acc:.4f}, Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}")

# Save the results DataFrame to a CSV file
results_df.to_csv("VanillaGPT2_high_features_results.csv", index=False)
