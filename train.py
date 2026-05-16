import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv("imdb_balanced_10k.csv")
texts = df["text"].astype(str).values
labels = df["label"].values
y = np.array(labels, dtype=np.float32)

texts_train, texts_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, random_state=42
)


tfidf = TfidfVectorizer(
    max_features=15000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.85
)
X_train = tfidf.fit_transform(texts_train).toarray()
X_test = tfidf.transform(texts_test).toarray()

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1,1).to(device)


class Model(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.relu(self.fc2(x))
        x = self.drop(x)
        return torch.sigmoid(self.fc3(x))

model = Model(X_train.shape[1]).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

print("Training start...")
epochs = 100

for epoch in range(epochs):
    model.train()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        acc = ((model(X_test) > 0.5) == y_test).float().mean().item()

    print(f"Epoch {epoch+1:2d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

print("\n Done!")