import torch as T
import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork_CartPole(nn.Module):
    '''
    Deep Q-Network for the CartPole environment.
    This network takes the state of the environment as input and outputs Q-values for each action.
    '''
    def __init__(self, input_dims: int, output_dims: int) -> None:
        '''
        Initializes the Deep Q-Network for the CartPole environment.
        :param input_dims: Number of input dimensions (state space size).
        :param output_dims: Number of output dimensions (action space size).
        '''
        super(DeepQNetwork_CartPole, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.fc1 = nn.Linear(self.input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dims)
        

    def forward(self, state: T.Tensor) -> T.Tensor:
        '''
        Forward pass through the network.
        :param state: Input state tensor.
        :return: Output tensor containing Q-values for each action.
        '''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions
