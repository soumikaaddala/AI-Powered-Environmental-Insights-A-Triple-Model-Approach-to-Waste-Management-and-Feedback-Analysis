import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from nltk.corpus import stopwords
import nltk

Download NLTK stopwords

nltk.download('stopwords')

Load dataset

print("Loading dataset...")
try:
df = pd.read_csv('IMDB-Dataset.csv', usecols=['review', 'sentiment'])
print("Dataset loaded successfully!")
except FileNotFoundError:
print("Error: 'IMDB-Dataset.csv' not found in the current directory!")
exit()

Clean text function

def clean_text(text):
if not isinstance(text, str):
return []
text = text.lower()
text = re.sub(r'[^a-zA-Z\s]', '', text)
stop_words = set(stopwords.words('english'))
words = [word for word in text.split() if word not in stop_words]
return words

print("Cleaning text data...")
df['cleaned_content'] = df['review'].apply(clean_text)
print("Text cleaning completed!")

Map Sentiment (0 -> Negative, 1 -> Positive)

df['label'] = df['sentiment'].map({'negative': 0, 'positive': 1})

Train Word2Vec Model

print("Training Word2Vec model...")
word2vec_model = Word2Vec(sentences=df['cleaned_content'], vector_size=300, window=5, min_count=5, workers=4)
word2vec_model.save('word2vec_model.model')
print("Word2Vec model saved!")

Convert Text to Word2Vec Embeddings

def text_to_embedding(text):
vectors = [word2vec_model.wv[word] for word in text if word in word2vec_model.wv]
if len(vectors) == 0:
return np.zeros(300)  # Return zero vector if no words found
return np.mean(vectors, axis=0)

print("Converting text to embeddings...")
df['embeddings'] = df['cleaned_content'].apply(text_to_embedding)

Prepare Data

X = np.array(df['embeddings'].tolist())
y = df['label'].values

Split Data

print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Define Dataset

class FeedbackDataset(Dataset):
def init(self, X, y):
self.X = torch.tensor(X, dtype=torch.float32)
self.y = torch.tensor(y, dtype=torch.long)
def len(self):
return len(self.y)
def getitem(self, idx):
return self.X[idx], self.y[idx]

Create DataLoader

train_data = FeedbackDataset(X_train, y_train)
test_data = FeedbackDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

Define GRU Model

class SentimentGRU(nn.Module):
def init(self, input_size=300, hidden_size=128, num_layers=2, output_size=2):
super(SentimentGRU, self).init()
self.hidden_size = hidden_size
self.num_layers = num_layers
self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
self.fc = nn.Linear(hidden_size, output_size)

def forward(self, x):  
    x = x.unsqueeze(1)  # Adding sequence dimension  
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  
    out, _ = self.gru(x, h0)  
    out = self.fc(out[:, -1, :])  
    return out

Initialize model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentGRU().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

Train Model

print("Training the GRU model...")
for epoch in range(10):
model.train()
total_loss = 0
for X_batch, y_batch in train_loader:
X_batch, y_batch = X_batch.to(device), y_batch.to(device)
optimizer.zero_grad()
outputs = model(X_batch)
loss = criterion(outputs, y_batch)
loss.backward()
optimizer.step()
total_loss += loss.item()
print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

Save Model

torch.save(model.state_dict(), 'sentiment_modelGRU.pth')
print("GRU model training completed and saved!")

Evaluate Model

print("Evaluating model on test data...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
for X_batch, y_batch in test_loader:
X_batch, y_batch = X_batch.to(device), y_batch.to(device)
outputs = model(X_batch)
_, predicted = torch.max(outputs, 1)
total += y_batch.size(0)
correct += (predicted == y_batch).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
