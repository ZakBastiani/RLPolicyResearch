import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical
import time
from expression_tree import ExpressionTree
import math


class BTSTransformerModel(nn.Module):
    def __init__(self, max_depth, two_children_funcs, one_children_funcs, variables, dev):
        super(BTSTransformerModel, self).__init__()
        self.device = dev
        self.max_depth = max_depth
        self.two_children_funcs = two_children_funcs
        self.two_children_num = len(two_children_funcs)
        self.one_children_funcs = one_children_funcs
        self.one_children_num = len(one_children_funcs) + len(two_children_funcs)
        self.variables = variables
        self.library_size = len(self.two_children_funcs) + len(self.one_children_funcs) + len(self.variables)
        self.input_size = 2 * (self.library_size + 1)

        # self.position = OneDimensionalPositionalEncoding(d_model=self.input_size, max_len=max_depth)
        self.position = TwoDimensionalPositionalEncoding(d_model=self.input_size, max_len=max_depth, device=self.device)
        self.transformer = nn.Transformer(d_model=self.input_size, nhead=1, num_encoder_layers=1, num_decoder_layers=1,
                                          batch_first=True)

        self.linear = nn.Linear(in_features=self.input_size, out_features=self.library_size)

        # self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=max_depth, num_layers=1, bias=True, batch_first=True,
        #                     dropout=0, bidirectional=False, proj_size=0)
        # self.linear = nn.Linear(in_features=self.max_depth, out_features=self.library_size)

        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

    def forward(self, x, p):
        x = self.position(x, p)
        x = self.transformer(x, x)
        # x, hidden = self.lstm(x, hidden)
        x = self.linear(x)
        x = self.softmax(x)
        return x

    def sample(self, n, device):
        sample_equs = {}
        oversampling_scalar = 1.5
        with torch.no_grad():
            i = 0
            dictionary = {"Fetch PS Time": 0, "Prediction Time": 0, "Apply Rules Time": 0,
                          "Rand_Cat Time": 0, "Add Node Time": 0, "Build Time": 0, "Comparison Time": 0}

            a = time.perf_counter()
            trees = ExpressionTree(int(oversampling_scalar * n), self.two_children_funcs, self.one_children_funcs, self.variables, self.max_depth, device)
            dictionary["Build Time"] += time.perf_counter() - a

            for j in range(self.max_depth):
                a = time.perf_counter()
                inputs = trees.get_inputs()
                positions = trees.get_positions()
                dictionary["Fetch PS Time"] += time.perf_counter() - a

                a = time.perf_counter()
                inputs = torch.squeeze(inputs, 1)
                x = self.forward(inputs.float().to(self.device), positions.float().to(self.device))
                x = x.to(device)
                dictionary["Prediction Time"] += time.perf_counter() - a

                a = time.perf_counter()
                rules = trees.fetch_rules(j)
                x = x * rules
                x = x / torch.sum(x, dim=2, keepdim=True)
                dictionary["Apply Rules Time"] += time.perf_counter() - a

                a = time.perf_counter()
                predicted_vals = Categorical(x).sample()
                dictionary["Rand_Cat Time"] += time.perf_counter() - a

                a = time.perf_counter()
                trees.add(predicted_vals.T[j].char(), j)
                # trees.add(x.squeeze(1).char(), j)
                dictionary["Add Node Time"] += time.perf_counter() - a

            a = time.perf_counter()
            equations = trees.equation_string()
            unique = []
            for index in range(int(oversampling_scalar * n)):
                equ = equations[index]
                if equ not in sample_equs:
                    unique.append(index)
                    sample_equs[equ] = -torch.inf
                    i += 1
                    if i == n:
                        break
                elif n - i >= oversampling_scalar * n - index:
                    unique.append(index)
                    i += 1

            trees.reduce(unique)
            dictionary["Comparison Time"] += time.perf_counter() - a
            # print(dictionary)
            return trees, dictionary, sample_equs


class OneDimensionalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[0, :x.size(1)]
        return x


class TwoDimensionalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 32, device=torch.device("cpu")):
        super().__init__()
        self.d_model = d_model
        self.d_model_mod_ceil = math.ceil(d_model/4)
        self.d_model_ceil = math.ceil(d_model/4) * 4
        self.max_length = max_len
        self.device = device
        self.div_term_10000 = torch.exp(torch.arange(0, self.d_model, 4) * (-math.log(10000.0) / self.d_model)).to(self.device)
        self.div_term_1 = torch.exp(torch.arange(0, self.d_model, 4) * (-math.log(1.0) / self.d_model)).to(self.device)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, sequence_length, embedding_dim]``
        """
        pe = torch.zeros((len(x), self.max_length, self.d_model_ceil), device=self.device)
        pe[:, :, 0::4] = torch.sin(positions[:, :, 0].unsqueeze(2).repeat(1, 1, self.d_model_mod_ceil) * self.div_term_10000)
        pe[:, :, 1::4] = torch.cos(positions[:, :, 0].unsqueeze(2).repeat(1, 1, self.d_model_mod_ceil) * self.div_term_10000)
        pe[:, :, 2::4] = torch.sin(positions[:, :, 1].unsqueeze(2).repeat(1, 1, self.d_model_mod_ceil) * self.div_term_1)
        pe[:, :, 3::4] = torch.cos(positions[:, :, 1].unsqueeze(2).repeat(1, 1, self.d_model_mod_ceil) * self.div_term_1)
        x = x + pe[:, :, :self.d_model]
        return x

