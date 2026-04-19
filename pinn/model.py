## Physics-Informed Neural Network — Model Architecture
## Feedforward neural network mapping spatial coordinates (x, y)
## to displacement field (u_x, u_y).
## All activations are tanh to support second-order autograd differentiation
## required by the PDE residual loss.

import torch
import torch.nn as nn
import numpy as np


class PINN(nn.Module):
    """
    Feedforward neural network representing the displacement field u(x, y).

    Input  : (x, y) — spatial coordinates, shape (N, 2)
    Output : (u_x, u_y) — displacement components, shape (N, 2)

    Architecture:
        Input layer  : 2 neurons
        Hidden layers: n_hidden layers of width n_neurons, tanh activation!!! -> tanh since the activation function must be double differentiable for the 
                       PDE residual loss (which involves second derivatives of the network output with respect to the input coordinates) 
        Output layer : 2 neurons, no activation (linear output)
    """

    def __init__(self, n_hidden=5, n_neurons=128):
        super().__init__()

        # ── Build layer list ─────────────────────────────────────────────────
        layers = []

        # Input → first hidden layer
        layers.append(nn.Linear(2, n_neurons))
        layers.append(nn.Tanh())

        # Hidden → hidden layers
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())

        # Last hidden → output layer (no activation — linear output)
        layers.append(nn.Linear(n_neurons, 2))

        self.network = nn.Sequential(*layers)

        # ── Weight initialisation ────────────────────────────────────────────
        self._initialise_weights()

    def _initialise_weights(self):
        """
        random weight initialisation matters for tanh networks. If weights are too large, 
        tanh saturates immediately and gradients vanish. If weights are too small, the network
        is effectively linear and cannot learn complex functions. Xavier initialisation sets the weight scale based 
        on the number of input and output neurons of each layer, keeping the variance of activations approximately 
        constant across layers at the start of training. This gives the optimiser a much better starting point.
        Xavier (Glorot) uniform initialisation for all linear layers.
        Keeps activation variance stable across layers at initialisation,
        which is important for smooth tanh networks.
        Biases initialised to zero.
        """
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, xy):
        """
        Forward pass.

        Parameters
        ----------
        xy : torch.Tensor, shape (N, 2)
            Spatial coordinates. Column 0 is x, column 1 is y.
            Must have requires_grad=True for autograd differentiation.

        Returns
        -------
        torch.Tensor, shape (N, 2)
            Predicted displacements. Column 0 is u_x, column 1 is u_y.
        """
        return self.network(xy)