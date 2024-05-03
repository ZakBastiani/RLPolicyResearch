from datetime import datetime
import pickle as pkl
import torch
from torch import nn
from policies import risk_seeking_policy
import time
from BTS_Transformer_model import BTSTransformerModel
import os
import json
import numpy as np
from expression_tree import simplify_equation

CUDA = torch.device('cuda')
CPU = torch.device('cpu')


def train(x, y, two_children_funcs=None, one_children_funcs=None, epochs=2000, max_depth=32, epsilon_init=0.05, entropy_coef=0.005,
          batch=1000, lr=5E-4, weight_decay=0, inference_mode=False, policy="Standard", opt_device=CPU, transformer_device=CUDA):

    if one_children_funcs is None:
        one_children_funcs = ["log", "sin", "cos", "sqrt", "exp"]
    if two_children_funcs is None:
        two_children_funcs = ["+", "-", "*", "/", "**"]

    one_children_funcs = ["np." + s for s in one_children_funcs]
    variables = [f"x[{i}]" for i in range(len(x))]
    variables += ["1", "const"]

    model = BTSTransformerModel(max_depth, two_children_funcs, one_children_funcs, variables, dev=transformer_device).to(transformer_device)
    opt = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    run_info = {
        "X": x.tolist(),
        "Y": y.tolist(),
        "two_children_funcs": two_children_funcs,
        "one_children_funcs": one_children_funcs,
        "variables": variables,
        "epochs": epochs,
        "max_depth": max_depth,
        "epsilon": epsilon_init,
        "entropy_coef": entropy_coef,
        "batch": batch,
        "date": str(datetime.now()),
        "inference_mode": inference_mode,
        "model_type": model.__class__.__name__,
        "model_parameters": {
            "max_depth": max_depth,
            "transformer_device": str(transformer_device)
        },
        "optimizer_type": opt.__class__.__name__,
        "optimizer_parameters": {
            "lr": lr,
            "weight_decay": weight_decay,
            "opt_device": str(opt_device)
            # Add more optimizer-specific parameters here if needed
        }
    }

    print_index = int(epochs/20) if int(epochs/20) > 1 else 1
    # initialize
    x = x.to(opt_device)
    y = y.to(opt_device)

    crit = torch.nn.CrossEntropyLoss(reduction="none")
    best_func = {"Equation": "", "Constants": [], "Loss": -torch.inf}
    start_time = time.perf_counter()
    cycle_time = time.perf_counter()
    epoch_info = {"Loss": [], "Cross Loss": [], "Entropy Loss": [], "Epoch Time": [], "Best Reward": [],
                  "Median Reward": [], "Baseline Reward": [], "Best Function": [], "Rewards": [],
                  "Expression Losses": [], "Full Entropy": [], "Node Counts": []}
    timer_dictionary = {"Sample Time": [], "Sample Time In-depth": [],
                        "Opt Time": [], "Reward": [], "Prediction": [], "Epoch Time": []}
    eq_dict = {}

    for i in range(epochs):
        if policy != 2:
            epsilon = epsilon_init
        else:
            epsilon = 0.3 + i * (epsilon_init - 0.3)/epochs
        # Sample from the model
        opt.zero_grad()
        a = time.perf_counter()
        trees, times, all_equations = model.sample(batch, opt_device)
        node_counts = trees.get_node_counts().to(CPU)
        epoch_info["Node Counts"].append(node_counts)
        timer_dictionary["Sample Time In-depth"].append(times)
        timer_dictionary["Sample Time"].append(time.perf_counter() - a)
        # optimize sampled functions
        a = time.perf_counter()
        # trees.bayes_opt(x, y, history=eq_dict, simplify_equ=False) if inference_mode else trees.opt(x, y)
        trees.opt(x, y, inference_mode)
        r_2s = trees.calc_r2s(x, y)
        for k, eq in enumerate(trees.equation_string()):
            eq_dict[eq] = {"Reward": trees.rewards[k], "Node Count": node_counts[k], "R2": r_2s[k]}
        epoch_info["Rewards"].append(trees.rewards)
        # epoch_info["Expression Losses"].append(losses)
        timer_dictionary["Opt Time"].append(time.perf_counter() - a)
        # Add a policy that evaluates only the top x%
        a = time.perf_counter()
        # epsilon_star = -0.25 * i/epochs + 0.3
        trees, baseline, policy_info = risk_seeking_policy(trees, epsilon, inference_mode)
        timer_dictionary["Reward"].append(time.perf_counter() - a)
        a = time.perf_counter()
        # Checks to see if a new best tree has been found
        best_ind = np.nanargmax(trees.rewards)
        if trees.rewards[best_ind] > best_func["Loss"]:
            best_func["Equation"] = trees.equation_string()[best_ind]
            best_func["Constants"] = trees.constants[best_ind].tolist()
            best_func["Loss"] = trees.rewards[best_ind]
            if inference_mode:
                best_func["Noise"] = trees.noise[best_ind].tolist()

        inputs = trees.get_inputs()
        positions = trees.get_positions()
        labels = trees.get_labels()

        # make prediction
        pred = model(inputs.float().to(transformer_device), positions.float().to(transformer_device)).to(opt_device)

        # Cross entropy loss
        if inference_mode:
            rewards = torch.tensor([reward for reward in trees.rewards], device=opt_device)
            # rewards = 1 / (1 + torch.exp(-rewards/1000)) - 1 / (1 + np.exp(-baseline/1000))
            cross_entropy_loss = torch.mean(crit(pred.permute(0, 2, 1).float(), labels.permute(0, 2, 1).float()).T)/25
            # cross_entropy_loss = torch.mean(rewards * crit(pred.permute(0, 2, 1).float(), labels.permute(0, 2, 1).float()).T)
        else:
            rewards = torch.tensor([reward - baseline for reward in trees.rewards], device=opt_device, dtype=torch.float64)
            if policy == 0 or policy == 2:
                cross_entropy_loss = torch.mean(rewards * crit(pred.permute(0, 2, 1).float(), labels.permute(0, 2, 1).float()).T)
            else:
                cross_entropy_loss = torch.mean(crit(pred.permute(0, 2, 1).float(), labels.permute(0, 2, 1).float()).T)/25

        entropy_loss = entropy_coef * torch.mean(torch.sum(pred.float() * torch.log(pred.float()), dim=2))
        # entropy_loss = entropy_coef * torch.sum(pred * torch.log(pred))/pred.shape[0]
        loss = cross_entropy_loss + entropy_loss
        if torch.is_complex(loss):
            print(rewards)
            for iterate, r in enumerate(trees.rewards):
                if torch.is_complex(r):
                    print(r)
                    print(trees.equation_string()[iterate])
            print("Complex Error")
        if torch.isnan(loss):
            print("Nan Loss")
        # Back propagate
        loss.backward()
        opt.step() 
        timer_dictionary["Prediction"].append(time.perf_counter() - a)
        timer_dictionary["Epoch Time"].append(time.perf_counter() - cycle_time)
        epoch_info["Loss"].append(loss.detach().item())
        epoch_info["Cross Loss"].append(cross_entropy_loss.detach().item())
        epoch_info["Entropy Loss"].append(entropy_loss.detach().item())
        epoch_info["Best Reward"].append(policy_info[0])
        epoch_info["Median Reward"].append(policy_info[1])
        epoch_info["Baseline Reward"].append(policy_info[2])
        epoch_info["Best Function"].append(trees.equation_string()[best_ind])
        if (i + 1) % print_index == 0:
            print(f"Cross Loss: {cross_entropy_loss}, Entropy: {entropy_loss}")
            print(f"Epoch: {i}, Loss: {loss.detach().item()}")
            print(f"Run time: {time.perf_counter()-start_time}, Epoch Time: {(time.perf_counter() - cycle_time)}")
        cycle_time = time.perf_counter()

    try:
        best_func["Simplified Equation"] = simplify_equation(best_func["Equation"], len(best_func["Constants"]))
    except:
        print("Failed to simplify final function")

    return {"Best Function": best_func, "Timings": timer_dictionary, "Iteration Info": epoch_info,
            "All Equations Tested": eq_dict, "Run Info": run_info}


def save_results(results, base_name, loc):
    count = 0
    file_name = f"{base_name}_{count}.pkl"

    # Check if the file already exists, incrementing count if necessary
    while os.path.exists(os.path.curdir + loc + file_name):
        count += 1
        file_name = f"{base_name}_{count}.pkl"

    with open(os.getcwd() + loc + file_name, 'wb') as file:
        pkl.dump(results, file)

    print(f"Dictionary has been saved as JSON in {file_name}")
