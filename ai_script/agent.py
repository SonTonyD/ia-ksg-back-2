import torch
import torch.nn as nn
import torch.optim as optim
import random

# Définition du modèle LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        if x.shape[0] != 1:
            # Si la forme de x est différente, calculez la moyenne sur la deuxième dimension (sequence_length)
            out = torch.mean(out, dim=1)
        else:
            # Sinon, prenez la dernière sortie de la séquence
            out = self.fc(out[:, -1, -1])
        return out

# Classe de l'agent DQN
class DQN_LSTM_Agent:
    def __init__(self, input_size, hidden_size, output_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel(input_size, hidden_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = []  # Expérience replay buffer
        self.batch_size = 32
        self.gamma = 0.99

    def select_action(self, state):
        # Epsilon-greedy exploration
        if random.random() < 0.1:
            return random.choice([0, 1])
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            q_values = self.model(state.unsqueeze(0)).squeeze(0)
            return q_values.argmax().item()

    def remember(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*minibatch)

        state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32, device=self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.int64, device=self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device)

        q_values = self.model(state_batch)
        next_q_values = self.model(next_state_batch)
        q_value = q_values.mean()
        next_q_value, _ = next_q_values.max(dim=0)
        expected_q_value = reward_batch + self.gamma * next_q_value * (1 - done_batch)

        loss = self.criterion(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_agent(self, path):
        torch.save(self.model.state_dict(), path)

    def load_agent(self, path):
        self.model.load_state_dict(torch.load(path))
