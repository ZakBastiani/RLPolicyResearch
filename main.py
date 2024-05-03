import math
import torch
import sympy
from expression_tree import ExpressionTree
from trainer import train, save_results
import os
import numpy as np
import json
import warnings
import pandas as pd
from pathlib import Path
import pickle as pkl
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
np.seterr(all='ignore')

with open(os.path.curdir + "\\functions\\nguyen.json", "r") as functions:
    nguyen = json.load(functions)

func = nguyen["12"]
x_0 = torch.linspace(0, 1, 20)
x_1 = torch.linspace(0, 1, 20)

x = torch.cat([x_0.repeat(len(x_1)).unsqueeze(0), x_1.unsqueeze(1).repeat((1, len(x_0))).flatten().unsqueeze(0)], dim=0)
y = eval(func["Function"])

info = train(x, y, epochs=2000, batch=1000, epsilon_init=0.05, policy=2, entropy_coef=0.005, lr=0.0005, max_depth=32, inference_mode=False)
info["Ground Truth"] = func
info["Ground Truth"]["X"] = x
info["Ground Truth"]["Y"] = y

save_results(info, "Nguyen_12_linear", "\\run_data\\")
