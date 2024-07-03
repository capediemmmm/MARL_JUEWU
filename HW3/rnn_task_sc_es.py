import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch.utils.data import random_split

# 数据生成
DATA_SIZE = 1000
sine_data_size = np.random.randint(int(0.3 * DATA_SIZE), int(0.7 * DATA_SIZE))
sigmoid_data_size = DATA_SIZE - sine_data_size
steps = np.arange(0, 10, 0.5)
sine_init = np.random.uniform(-3, 3, (sine_data_size, 2))
sine_data = np.sin(sine_init[:, :1] * steps + sine_init[:, 1:])
sigmoid_init = np.random.uniform(-3, 3, (sigmoid_data_size, 2))
sigmoid_data = 1 / (1 + np.exp(0 - sigmoid_init[:, :1] * steps + sigmoid_init[:, 1:]))
sine_data = np.concatenate((sine_data, np.ones((sine_data_size, 1))), axis=1)
sigmoid_data = np.concatenate((sigmoid_data, np.zeros((sigmoid_data_size, 1))), axis=1)
data = np.concatenate((sine_data, sigmoid_data), axis=0)
data = torch.Tensor(data)
train_set, test_set = random_split(data, [0.8, 0.2])

# 定义网络
class SimpleClassificationRNN(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleClassificationRNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True, num_layers=1)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, seq, hc=None):
        tmp, hc = self.rnn(seq, hc)
        out = torch.sigmoid(self.linear(hc[-1, :, :]))
        return out, hc

hidden_size = 16
learning_rate = 0.01
model = SimpleClassificationRNN(hidden_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

def cal_accuracy(preds, true_values):
    preds = torch.where(preds > 0.5, 1, 0)
    acc = torch.sum(1 - torch.abs(preds - true_values)) / preds.shape[0]
    return acc

def cal_f1(preds, true_values):
    preds = torch.where(preds > 0.5, 1, 0).cpu().numpy()
    true_values = true_values.cpu().numpy()
    f1 = f1_score(true_values, preds)
    return f1

# 训练与早停策略
epochs = 500
loss_log = []
best_acc = 0
patience = 20
counter = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output, _ = model(train_set[:][:, :-1, np.newaxis])
    loss = criterion(output.view(-1), train_set[:][:, -1])
    acc = cal_accuracy(output.view(-1), train_set[:][:, -1])
    loss.backward()
    optimizer.step()
    
    # 调度器
    scheduler.step(loss)
    
    if epoch % 10 == 0:
        print("Epoch {}: loss {} acc {}".format(epoch, loss.item(), acc))
    
    # 早停策略
    model.eval()
    output, _ = model(test_set[:][:, :-1, np.newaxis])
    test_loss = criterion(output.view(-1), test_set[:][:, -1])
    test_acc = cal_accuracy(output.view(-1), test_set[:][:, -1])
    
    if test_acc > best_acc:
        best_acc = test_acc
        counter = 0
    else:
        counter += 1
    
    if counter >= patience:
        print("Early stopping at epoch {}".format(epoch))
        break

# 测试集上的表现
output, _ = model(test_set[:][:, :-1, np.newaxis])
loss = criterion(output.view(-1), test_set[:][:, -1])
acc = cal_accuracy(output.view(-1), test_set[:][:, -1])
f1 = cal_f1(output.view(-1), test_set[:][:, -1])

print("Test set: loss {} acc {} f1 {}".format(loss.item(), acc, f1))
