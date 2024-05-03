import math
import numpy as np
import torch
from scipy import optimize
from torch.nn import functional as F
from sympy import *
import signal


np.seterr(divide='ignore', invalid='ignore')


class ExpressionTree:
    def __init__(self, n, two_children_funcs, one_children_funcs, variables, max_depth, device):
        self.n = n
        self.max_dataset_size = 1000
        self.max_depth = max_depth
        self.library = two_children_funcs + one_children_funcs + variables
        self.library_size = len(self.library)
        self.two_children_num = len(two_children_funcs)
        self.one_children_num = len(two_children_funcs) + len(one_children_funcs)
        self.input_size = 2 * (len(self.library) + 1)
        self.device = device
        self.empty = torch.zeros((1, 1, self.input_size), device=self.device, dtype=torch.bool)
        self.roots = [Node(Node(None, False, -1), False, 0) for i in range(n)]
        self.constants = [np.random.rand(1)] * n
        self.noise = [np.random.rand(1)] * n
        self.rewards = [0] * n
        self.inputs_backlog = torch.zeros((self.n, max_depth + 2, 2), device=self.device, dtype=torch.int8)
        self.bforder = torch.zeros((self.n, max_depth), device=self.device, dtype=torch.int8)
        self.valid_nodes = torch.zeros((self.n, max_depth + 2), device=self.device, dtype=torch.bool)
        self.positions = torch.zeros((self.n, max_depth + 2, 2), device=self.device, dtype=torch.float64)
        self.incremental_constant = [0] * n
        self.node_order = [[self.roots[i]] for i in range(n)]
        self.node_counts = torch.zeros(self.n, device=device)
        self.rules = torch.ones((n, max_depth + 2, self.library_size), device=self.device, dtype=torch.bool)
        # Rules for the expression tree
        self.constants_rule = torch.ones(self.library_size, device=self.device, dtype=torch.bool)
        self.ONF_rule = torch.ones(self.library_size, device=self.device, dtype=torch.bool)
        self.one_func_or_vars_rule = torch.ones(self.library_size, device=self.device, dtype=torch.bool)
        self.vars_rule = torch.ones(self.library_size, device=self.device, dtype=torch.bool)
        self.constants_rule[-1] = 0.0
        for j in range(self.two_children_num, self.one_children_num):
            self.ONF_rule[j] = 0.0
        for j in range(self.two_children_num):
            self.one_func_or_vars_rule[j] = 0.0
        for j in range(self.one_children_num):
            self.vars_rule[j] = 0.0

    # Evaluates x for the expression tree
    def evaluate(self, x):
        equations = self.equation_string()
        y = []
        for i in range(self.n):
            self.incremental_constant[i] = 0
            c = self.constants[i]
            y.append(eval(equations[i]))
        return y

    def equation_string(self):
        equations = []
        for i, r in enumerate(self.roots):
            self.incremental_constant[i] = 0
            equation, node_num = self.equation_string_rec(r, i, 0)
            equations.append(equation)
            self.node_counts[i] = node_num
        return equations

    def equation_string_rec(self, current, index, node_count):
        node_count += 1
        if current.right is not None:
            left, node_count = self.equation_string_rec(current.left, index, node_count)
            right, node_count = self.equation_string_rec(current.right, index, node_count)
            return "(" + left + self.library[current.val] + right + ")", node_count
        elif current.left is not None:
            left, node_count = self.equation_string_rec(current.left, index, node_count)
            return "(" + self.library[current.val] + "(" + left + "))", node_count
        else:
            if current.val == len(self.library) - 1:
                char = f"c[{self.incremental_constant[index]}]"
                self.incremental_constant[index] += 1
            else:
                char = self.library[current.val]
            return char, node_count

    # Adds the node to the next valid location in the expression tree according to breadth first traversal
    def add(self, val, node_num):
        self.bforder[:, node_num] = val
        self.valid_nodes[:, node_num] = torch.tensor([node_num < len(self.node_order[i]) for i in range(self.n)],
                                                     device=self.device)
        bools = (self.one_children_num > val) * (val >= self.two_children_num)
        self.rules[:, node_num] *= ~((bools.float().unsqueeze(1) @ (~self.ONF_rule).float().unsqueeze(0)).bool())
        has_siblings = []
        node_numbers = torch.zeros(self.n, dtype=torch.long)
        for i in range(self.n):
            if node_num >= len(self.node_order[i]):
                has_siblings.append(False)
                node_numbers[i] = self.max_depth - 1
                continue
            new_node = self.node_order[i][node_num]
            node_numbers[i] = len(self.node_order[i])
            new_node.val = val[i].item()
            has_siblings.append(self.node_order[i][node_num].has_sibling)
            if val[i] < self.two_children_num:
                new_node.left = Node(new_node, True, len(self.node_order[i]) + 1)
                self.node_order[i].append(new_node.left)
                new_node.right = Node(new_node, False, len(self.node_order[i]) + 2)
                self.node_order[i].append(new_node.right)
            elif val[i] < self.one_children_num:
                new_node.left = Node(new_node, False, len(self.node_order[i]) + 1)
                self.node_order[i].append(new_node.left)

        if node_num < self.max_depth - 1:
            self.inputs_backlog[torch.arange(self.n), node_num + 1, 1] = (val + 1) * torch.tensor(has_siblings,
                                                                                                  device=self.device)
            self.inputs_backlog[torch.arange(self.n), node_numbers, 0] = val + 1
            self.rules[torch.arange(self.n), node_numbers] = self.rules[:, node_num]
            self.positions[:, node_numbers, 0] = self.positions[:, node_num, 0] + 1
            self.positions[:, node_numbers, 1] = self.positions[:, node_num, 1] - 1 / torch.pow(2, self.positions[:,
                                                                                                   node_num, 0] + 1)
        if node_num < self.max_depth - 2:
            node_numbers += 1
            self.inputs_backlog[torch.arange(self.n), node_numbers, 0] = (val + 1)
            self.rules[torch.arange(self.n), node_numbers] = self.rules[:, node_num]
            self.positions[:, node_numbers, 0] = self.positions[:, node_num, 0] + 1
            self.positions[:, node_numbers, 1] = self.positions[:, node_num, 1] + 1 / torch.pow(2, self.positions[:,
                                                                                                   node_num, 0] + 1)
        # Need to add the sibling information and I need to read the code to see if rules is working correctly

    # Returns the preorder_traversal
    def get_labels(self):
        return F.one_hot(self.bforder.long(), num_classes=self.library_size) * self.valid_nodes[:, :self.max_depth].unsqueeze(2)

    # Returns the parent sibling inputs that were used to generate the set
    def get_inputs(self):
        parents = F.one_hot(self.inputs_backlog[:, :self.max_depth, 0].long(),
                            num_classes=self.library_size + 1) * self.valid_nodes[:, :self.max_depth].unsqueeze(2)
        siblings = F.one_hot(self.inputs_backlog[:, :self.max_depth, 1].long(),
                             num_classes=self.library_size + 1) * self.valid_nodes[:, :self.max_depth].unsqueeze(2)
        return torch.cat([parents, siblings], dim=2).bool()

    def get_positions(self):
        return self.positions[:, :self.max_depth, :]

    # Fetches the parent and sibling values for the input node_num
    def fetch_ps(self, node_num):
        parents = F.one_hot(self.inputs_backlog[:, node_num, 0].long(), num_classes=self.library_size + 1).unsqueeze(1)
        siblings = F.one_hot(self.inputs_backlog[:, node_num, 1].long(), num_classes=self.library_size + 1).unsqueeze(1)
        return torch.cat([parents, siblings], dim=2)

    # Get the number of nodes in each expression tree
    def get_node_counts(self):
        return self.node_counts

    # Solves for the values for all of the constants in the expression tree
    def opt(self, x, y, inference_mode):
        x_full = x.cpu().numpy()
        y_full = y.cpu().numpy()
        data_set_size = len(y)
        std = np.std(y_full)
        equations = self.equation_string()
        for i in range(self.n):
            if data_set_size > self.max_dataset_size:
                perm = np.random.permutation(data_set_size)[:self.max_dataset_size]
                x = x_full.T[perm].T
                y = y_full[perm]
            else:
                x = x_full
                y = y_full
            try:
                # Checking to see if there are no constants
                if self.incremental_constant[i] == 0:
                    if inference_mode:
                        v = np.mean((eval(equations[i]) - y) ** 2)
                        self.noise[i] = v
                        c = []
                        self.rewards[i] = BIC_np_calc_loss(c, v, self.incremental_constant[i], self.node_counts[i], x_full, y_full, equations[i])
                    else:
                        self.rewards[i] = reward_func(eval(equations[i]), y, std)

                    if np.isnan(self.rewards[i]) or np.iscomplex(self.rewards[i]):
                        self.rewards[i] = np.nan
                    continue

                self.constants[i] = np.random.rand(self.incremental_constant[i])

                # There is a negative sign on the return change the reward for being maximized to being minimized
                def func(c):
                    nonlocal x
                    return y - eval(equations[i])

                info = optimize.least_squares(func, self.constants[i], method='lm')
                self.constants[i] = info.x
                c = self.constants[i]
                if inference_mode:
                    v = np.mean((eval(equations[i]) - y) ** 2)
                    self.noise[i] = v
                    self.rewards[i] = BIC_np_calc_loss(c, v, self.incremental_constant[i], self.node_counts[i], x_full, y_full, equations[i])
                else:
                    self.rewards[i] = reward_func(eval(equations[i]), y, std)
            except(ZeroDivisionError, ValueError):
                self.rewards[i] = np.nan

        if inference_mode:
            self.rewards = -np.nan_to_num(self.rewards, nan=np.inf)
        else:
            self.rewards = np.nan_to_num(self.rewards, nan=0)

        return self.rewards

    def calc_r2s(self, x, y):
        r_2s = []
        mu = torch.mean(y)
        normalizer = torch.sum((y - mu)**2)
        equations = self.equation_string()
        for i, equation in enumerate(equations):
            if self.rewards[i] == -torch.inf:
                r_2s.append(-np.inf)
                continue
            try:
                c = self.constants[i]
                device = self.device
                pred_y = eval(equation)
                r2 = calc_r_squared(pred_y, y, normalizer)
                if np.isnan(r2):
                    r_2s.append(-np.inf)
                else:
                    r_2s.append(r2)
            except:
                r_2s.append(-np.inf)
        return r_2s

    def fetch_rules(self, node_num):
        # Need to add constant rule here
        for i in range(self.n):
            if self.max_depth == len(self.node_order[i]):
                self.rules[i][node_num] *= self.vars_rule
            elif self.max_depth - 1 == len(self.node_order[i]):
                self.rules[i][node_num] *= self.one_func_or_vars_rule
            elif self.max_depth < len(self.node_order[i]):
                print("ERRR")
        return self.rules[:, node_num].unsqueeze(1)

    def reduce(self, indices):
        self.n = len(indices)

        self.roots = [self.roots[i] for i in indices]
        self.rewards = [self.rewards[i] for i in indices]
        self.constants = [self.constants[i] for i in indices]
        self.noise = [self.noise[i] for i in indices]
        self.node_order = [self.node_order[i] for i in indices]
        self.incremental_constant = [self.incremental_constant[i] for i in indices]

        indices_tensor = torch.tensor(indices.copy(), device=self.device, dtype=torch.int32)
        self.bforder = torch.index_select(self.bforder, dim=0, index=indices_tensor)
        self.inputs_backlog = torch.index_select(self.inputs_backlog, dim=0, index=indices_tensor)
        self.positions = torch.index_select(self.positions, dim=0, index=indices_tensor)
        self.node_counts = torch.index_select(self.node_counts, dim=0, index=indices_tensor)
        self.valid_nodes = torch.index_select(self.valid_nodes, dim=0, index=indices_tensor)
        self.rules = torch.index_select(self.rules, dim=0, index=indices_tensor)
        return


class Node:
    def __init__(self, parent, has_sibling, node_num):
        self.left = None
        self.right = None
        self.parent = parent
        self.sibling = None
        self.has_sibling = has_sibling
        self.node_num = node_num


def reward_func(pred_y, y, y_std):
    NRMSE = np.sqrt(np.mean((pred_y - y) ** 2)) / y_std
    return 1 / (1 + NRMSE)


def calc_r_squared(pred_y, y, normalizer):
    return 1 - torch.sum((y - pred_y)**2)/normalizer


def BIC_np_calc_loss(c, v, var_count, node_count, x, y, equation):
    # invert noise
    sample_size = len(x[0])
    s_x = eval(equation)
    loss = 0.5 * sample_size * np.log(v) + 0.5 * sample_size * math.log(2 * math.pi) + 0.5 * np.sum((y - s_x)**2)/v
    return (var_count + node_count) * math.log(sample_size) + 2 * loss


def simplify_equation(equation, c_count, x_count=10):
    symbols(f'c:{c_count}')
    for k in range(c_count):
        equation = equation.replace(f"c[{k}]", f"c{k}")
    symbols('x:10')
    for j in range(x_count):
        equation = equation.replace(f"x[{j}]", f"x{j}")
    equation = equation.replace("torch.tensor(1, device=device)", "1").replace("torch.", "").replace("np.", "")
    equation = str(simplify(equation))
    for k in range(c_count):
        equation = equation.replace(f"c{k}", f"c[{k}]")
    for j in range(x_count):
        equation = equation.replace(f"x{j}", f"x[{j}]")
    equation = equation.replace("sin", "torch.sin").replace("log", "torch.log").replace("cos", "torch.cos")
    return equation
