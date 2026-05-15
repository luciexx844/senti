import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ----------------------
# 加载 GloVe 词向量
# ----------------------
with open("tiny_glove.json", "r", encoding="utf-8") as f:
    glove = json.load(f)
embedding_dim = 50

# ----------------------
# 加载数据集
# ----------------------
df = pd.read_csv("imdb_balanced_10k.csv")

if "review" in df.columns:
    texts = df["review"].tolist()
else:
    texts = df.iloc[:, 0].tolist()

if "sentiment" in df.columns:
    labels = df["sentiment"].map({"positive": 1, "negative": 0}).values
else:
    labels = df.iloc[:, 1].values

# ----------------------
# 文本预处理
# ----------------------
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------
# GloVe 嵌入矩阵
# ----------------------
word_index = tokenizer.word_index
embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word_index.items():
    if i < max_words and word in glove:
        embedding_matrix[i] = glove[word]

# ----------------------
# 神经网络模型
# ----------------------
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(max_words, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix).float())
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(50, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.sigmoid(self.fc(h[-1]))

model = Model()

# ----------------------
# 训练
# ----------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train = torch.tensor(X_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print("Training start...")
for epoch in range(4):
    model.train()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        pred = model(X_test).squeeze()
        acc = accuracy_score(y_test.numpy(), (pred > 0.5).numpy())
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

print("\nTraining finished successfully!")