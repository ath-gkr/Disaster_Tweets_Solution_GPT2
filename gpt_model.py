
import torch
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import yaml	
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


import sys

from typing import Union, List



class GPT2Classifier(nn.Module):
	def __init__(self, gpt2_model, num_classes=2):
		super(GPT2Classifier, self).__init__()
		self.gpt2 = gpt2_model
		self.gpt2.requires_grad_(False)  # Freeze the parameters of the GPT-2 model
		# self.dense = nn.Linear(gpt2_model.config.n_embd, num_classes)   #When it is just one layer 
		self.batchnorm1 = nn.BatchNorm1d(gpt2_model.config.n_embd)
		self.dense1 = nn.Linear(gpt2_model.config.n_embd, 256)
		self.batchnorm2 = nn.BatchNorm1d(256)
		self.dense2 = nn.Linear(256, 128)
		self.batchnorm3 = nn.BatchNorm1d(128)
		self.dense3 = nn.Linear(128, num_classes)
		# self.parameters = {}
		
	def forward(self, input_ids, attention_mask=None):
		outputs = self.gpt2(input_ids, attention_mask=attention_mask)
		hidden_states = outputs.last_hidden_state
		pooled_output = hidden_states[:, -3]  # Use the last token's hidden state
		x = self.batchnorm1(pooled_output)
		x = self.dense1(x)
		x = self.batchnorm2(x)
		x = self.dense2(x)
		x = self.batchnorm3(x)
                logits = self.dense3(x)
		#logits = self.dense(pooled_output)  # When it is just one layer
		return logits


class DisasterTweetDataset(Dataset):
	def __init__(self, texts, labels, tokenizer, max_length=64):
		
		self.texts = texts
		self.labels = labels
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, idx):
		text = self.texts[idx]
		label = self.labels[idx]
		inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
		inputs['input_ids'] = inputs['input_ids'].squeeze()
		inputs['attention_mask'] = inputs['attention_mask'].squeeze()
		inputs['label'] = torch.tensor(label, dtype=torch.long)
		return inputs

class TrainingAlg():

	def __init__(self, num_epochs=5, batch_size=64, experiment_folder="log/run_name/",epochs_to_save: Union[int, List[int]] = 1):

		#tensorboard logger:
		self.experiment_folder=experiment_folder
		self.writer = SummaryWriter(self.experiment_folder)

		#Create classifiers:
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.classifier = self.get_classifier()
		self.classifier.to(self.device)

		# Create the DataLoaders
		train_dataset, val_dataset = self.load_dataset()
		self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


		self.loss_function = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-5)


		if type(epochs_to_save) is int:
			if epochs_to_save==0:
				epochs_to_save=[]
			elif epochs_to_save<num_epochs and epochs_to_save>=0:
				#save 'epochs_to_save' nr of entries, and make sure that the last epoch is saved:
				save_frequency = num_epochs//epochs_to_save
				save_start = (num_epochs%epochs_to_save)-1+save_frequency
				epochs_to_save = [save_frequency*i+save_start for i in range(epochs_to_save)]
			else:
				epochs_to_save = [i for i in range(num_epochs)]
		self.epochs_to_save = epochs_to_save
		self.num_epochs = num_epochs
		print(self.epochs_to_save)

	def load_dataset(self, test_size=0.2):
		# Read the CSV file
		data = pd.read_csv("./data/train.csv")

		# Extract the text and target columns
		texts = data["text"].tolist()
		targets = data["target"].tolist()

		train_texts, val_texts, train_targets, val_targets = train_test_split(texts, targets, test_size=test_size, random_state=42)

		# Save validation data to a CSV file
		validation_data = pd.DataFrame({"text": val_texts, "target": val_targets})
		validation_data.to_csv(f"{self.experiment_folder}/validation_data.csv", index=False)

		# Create the training and validation Dataset instances
		train_dataset = DisasterTweetDataset(train_texts, train_targets, self.tokenizer)
		val_dataset = DisasterTweetDataset(val_texts, val_targets, self.tokenizer)
		return (train_dataset, val_dataset)

	def validate_dataset(self):
		# Load validation data from the CSV file
		validation_data = pd.read_csv(f"{self.experiment_folder}/validation_data.csv")
		val_texts = validation_data["text"].tolist()
		val_targets = validation_data["target"].tolist()

		val_dataset = DisasterTweetDataset(val_texts, val_targets, self.tokenizer)
		val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

		self.classifier.eval()
		mistakes = []
		with torch.no_grad():
			for batch in val_dataloader:
				input_ids, attention_mask, labels = batch
				input_ids = input_ids.to(self.device)
				attention_mask = attention_mask.to(self.device)
				labels = labels.to(self.device)

				outputs = self.classifier(input_ids, attention_mask=attention_mask)
				_, preds = torch.max(outputs.logits, 1)

				for i in range(len(labels)):
					if preds[i] != labels[i]:
						mistakes.append((val_texts[i], preds[i].item(), labels[i].item()))

		for text, predicted, actual in mistakes:
			print(f"Text: {text}\nPredicted: {predicted}\nActual: {actual}\n")

		return mistakes

	def get_classifier(self):
		self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
		self.tokenizer.pad_token = self.tokenizer.eos_token
		config = GPT2Config.from_pretrained("gpt2")
		self.model = GPT2Model.from_pretrained("gpt2", config=config)
		classifier = GPT2Classifier(self.model)
		
		return classifier

	def train(self):
		

		# Create an empty DataFrame to store the results
		results_df = pd.DataFrame(columns=["Epoch", "Train Loss", "Val Loss", "Accuracy"])
		
		#training loop
		for epoch in range(self.num_epochs):
			self.classifier.train()
			train_loss = 0.0

			for i, batch in enumerate(tqdm(self.train_dataloader)):
				input_ids = batch['input_ids'].to(self.device)
				attention_mask = batch['attention_mask'].to(self.device)
				labels = batch['label'].to(self.device)

				# Forward pass
				logits = self.classifier(input_ids, attention_mask=attention_mask)

				# Compute the loss
				loss = self.loss_function(logits, labels)

				# Backpropagation and optimization
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				train_loss += loss.item()
				#print(i)

			# Calculate average training loss for this epoch
			avg_train_loss = train_loss / len(self.train_dataloader)

			# Evaluate the model on the validation set
			self.classifier.eval()
			val_loss = 0.0
			correct_predictions = 0

			with torch.no_grad():
				for batch in self.val_dataloader:
					input_ids = batch['input_ids'].to(self.device)
					attention_mask = batch['attention_mask'].to(self.device)
					labels = batch['label'].to(self.device)

					logits = self.classifier(input_ids, attention_mask=attention_mask)
					loss = self.loss_function(logits, labels)
					val_loss += loss.item()

					# Calculate the number of correct predictions
					_, preds = torch.max(logits, dim=1)
					correct_predictions += (preds == labels).sum().item()

			# Calculate average validation loss and accuracy for this epoch
			avg_val_loss = val_loss / len(self.val_dataloader)
			val_accuracy = correct_predictions / len(self.val_dataloader)

			
			# Log the training loss
			self.writer.add_scalar('Loss/train', train_loss, epoch)

			# Log the validation loss
			self.writer.add_scalar('Loss/val', val_loss, epoch)

			# Log the validation accuracy
			self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)

			if epoch in self.epochs_to_save:
				model_name = self.experiment_folder+"model_{}.pt".format(epoch)
				torch.save(self.model.state_dict(), model_name)


			print(f"Epoch {epoch+1}/{self.num_epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Accuracy={val_accuracy:.4f}")
			# Append the results to the DataFrame
			results_df = results_df.append({
				"Epoch": epoch + 1,
				"Train Loss": avg_train_loss,
				"Val Loss": avg_val_loss,
				"Accuracy": val_accuracy
				}, ignore_index=True)

		# Save the results DataFrame to a CSV file
		results_df.to_csv("ModGPT2_results.csv", index=False)




if __name__ == "__main__":
	if len(sys.argv) > 1:
		config_file = sys.argv[1]
		with open(config_file, 'r') as file:
			config = yaml.safe_load(file)
		alg = TrainingAlg(**config)
	else:
		print("No configuration file provided\nusing default arguments instead")
		alg = TrainingAlg()
	
	alg.train()
