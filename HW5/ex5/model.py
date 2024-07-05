
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        '''
        TODO:
        conv1: 输入维度，输出维度:32，kernal_size:3, stride:2, padding:1
        conv2:输入维度：32，输出维度:32，kernal_size:3, stride:2, padding:1
        conv3:输入维度：32，输出维度:32，kernal_size:3, stride:2, padding:1
        conv4:输入维度：32，输出维度:32，kernal_size:3, stride:2, padding:1
        lstm:(LSTMCell)，输入维度：32*6*6，输出维度512
        critic：输入512，输出1
        actor：输入512，输出action_space
        '''
        # 定义卷积层
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        
        # 计算 LSTM 的输入维度，假设输入图像的初始大小为 (1, 84, 84)
        self.lstm_input_size = 32 * 6 * 6
        
        # 定义 LSTM 层
        self.lstm = nn.LSTMCell(self.lstm_input_size, 512)
        
        # 定义 actor 和 critic 网络
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):

        '''
        TODO: 完成前向过程
        要求：返回actor值，critic值，hx值，cx值
        '''
        # 前向传播
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # 将卷积层的输出展平为一维向量
        x = x.view(x.size(0), -1)
        
        # LSTM 前向传播
        hx, cx = self.lstm(x, (hx, cx))
        
        # 计算 critic 和 actor 值
        critic_value = self.critic(hx)
        actor_value = self.actor(hx)
        
        return actor_value, critic_value, hx, cx



