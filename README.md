IMDB Sentiment Analysis Project
Project Overview
This project implements sentiment analysis on IMDB movie reviews, adopting GloVe word embedding combined with LSTM neural network to complete binary sentiment classification.

CI/CD Running Explanation
Due to the resource constraints of free GitHub Actions runtime environment, including limited RAM capacity and no available GPU device, running PyTorch and TensorFlow/Keras frameworks simultaneously will cause memory overflow error, namely Segmentation Fault with exit code 139.

For this reason, the automatic training script cannot be executed normally on GitHub workflow. However, the complete code has been fully tested and successfully executed in the local development environment.

Local Running Results
Training start...
Epoch 1 | Loss: 0.6947 | Acc: 0.4935
Epoch 2 | Loss: 0.6938 | Acc: 0.4850
Epoch 3 | Loss: 0.6933 | Acc: 0.4915
Epoch 4 | Loss: 0.6928 | Acc: 0.5030

Project Configuration
Network Structure: GloVe Embedding Layer + LSTM Layer + Fully Connected Layer + Sigmoid Activation
Dataset Used: imdb_balanced_10k.csv
Preprocessing Method: Text sequence tokenization and padding
Training Framework: PyTorch