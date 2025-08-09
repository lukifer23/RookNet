from abc import ABC, abstractmethod

import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all chess models in this project.
    Ensures that all models adhere to a consistent interface for training and evaluation.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor representing the board state.

        Returns:
            A tuple containing the value and policy outputs.
            - value (torch.Tensor): A scalar tensor evaluating the position.
            - policy (torch.Tensor): A tensor representing the policy for all possible moves.
        """
        pass
