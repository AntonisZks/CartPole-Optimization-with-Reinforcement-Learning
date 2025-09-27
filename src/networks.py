import torch as T
import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork_CartPole(nn.Module):
    def __init__(self, input_dims: int, output_dims: int) -> None:
        super(DeepQNetwork_CartPole, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.fc1 = nn.Linear(self.input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, self.output_dims)
          

    def forward(self, state: T.Tensor) -> T.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)

        return actions
