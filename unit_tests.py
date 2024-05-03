import unittest
import torch
from expression_tree import ExpressionTree, simplify_equation
import os
import json
from torch.nn import functional as F
from policies import risk_seeking_policy


class TestExpressionTree(unittest.TestCase):
    def setUp(self):
        # Initialize any common resources or setup needed for the tests
        with open(os.path.curdir + "\\..\\functions\\nguyen.json", "r") as functions:
            nguyen = json.load(functions)
        func = nguyen["1"]
        # Sample the function n times
        x = eval(func["Dataset"]).unsqueeze(0)
        y = eval(func["Function"])
        func["Vars"].insert(1, "1")
        # train and return function
        self.tcf = func["TCF"]
        self.ocf = func["OCF"]
        self.vars = func["Vars"]
        self.library_size = len(func["Vars"]) + len(func["OCF"]) + len(func["TCF"])
        self.n = 2

        pass

    def tearDown(self):
        # Clean up after each test
        pass

    def test_add_node(self):
        # Initialize ExpressionTree with dummy values
        tree = ExpressionTree(self.n, self.tcf, self.ocf, self.vars, 32, torch.device('cpu'))

        # Call the add method with specific values
        tree.add(torch.tensor([1, 0], dtype=torch.int8), 0)  # Adjust the parameters based on your add method signature
        preorder = torch.zeros(32)
        preorder[0] = 1
        # Perform assertions to check if the tree is updated as expected
        for j in range(32):
            self.assertEqual(tree.bforder[0][j], preorder[j])
        self.assertEqual(3 == len(tree.node_order[0]), True)

    def test_add_complete_tree(self):
        # Initialize ExpressionTree with dummy values
        tree = ExpressionTree(self.n, self.tcf, self.ocf, self.vars, 32, torch.device('cpu'))

        # Add nodes to complete the tree
        zeros = torch.zeros((2, 28), dtype=torch.int8)
        bforder = torch.tensor([[0, 1], [6, 7], [9, 9], [9, 9]], dtype=torch.int8).T
        temp = torch.cat([bforder, zeros], dim=1)
        for i in range(32):
            tree.add(temp.T[i], i)
        # Perform assertions to check if the tree is updated as expected
        bforder = F.one_hot(bforder.long(), num_classes=self.library_size)
        zeros = torch.zeros((2, 28, self.library_size), dtype=torch.int8)
        bforder = torch.cat([bforder, zeros], dim=1)
        labels = tree.get_labels()
        for i in range(2):
            for j in range(32):
                for k in range(len(labels[i][j])):
                    self.assertEqual(labels[i][j][k], bforder[i][j][k])
        self.assertEqual(4 == len(tree.node_order[0]), True)

    def test_fetch_ps(self):
        # Initialize ExpressionTree with dummy values
        tree = ExpressionTree(self.n, self.tcf, self.ocf, self.vars, 32, torch.device('cpu'))

        # Add nodes to complete the tree
        zeros = torch.zeros((2, 28), dtype=torch.int8)
        preorder = torch.tensor([[0, 1], [6, 9], [9, 7], [9, 9]], dtype=torch.int8).T
        preorder = torch.cat([preorder, zeros], dim=1)
        for i in range(32):
            tree.add(preorder.T[i], i)

        inputs = torch.zeros((2, 32, tree.input_size), dtype=torch.bool)
        inputs[0][0][0] = 1
        inputs[0][0][tree.library_size+1] = 1
        inputs[0][1][1+0] = 1
        inputs[0][1][tree.library_size+1] = 1
        inputs[0][2][1+0] = 1
        inputs[0][2][1+6+tree.library_size+1] = 1
        inputs[0][3][1+6] = 1
        inputs[0][3][tree.library_size+1] = 1

        inputs[1][0][0] = 1
        inputs[1][0][tree.library_size+1] = 1
        inputs[1][1][1+1] = 1
        inputs[1][1][tree.library_size+1] = 1
        inputs[1][2][1+1] = 1
        inputs[1][2][1+9+tree.library_size+1] = 1
        inputs[1][3][1+7] = 1
        inputs[1][3][tree.library_size+1] = 1

        fetched_inputs = tree.get_inputs()
        # print(fetched_inputs)
        # print(inputs)
        for i in range(2):
            for j in range(32):
                for k in range(tree.input_size):
                    self.assertEqual(fetched_inputs[i][j][k], inputs[i][j][k])

    # Add more test methods for other functionalities of ExpressionTree
    def test_rules(self):
        one_children_funcs = ["torch.log", "torch.sqrt", "torch.sin", "torch.cos", "torch.sqrt"]
        tree = ExpressionTree(self.n, self.tcf, one_children_funcs, self.vars, 32, torch.device('cpu'))

        # Add nodes to complete the tree
        zeros = torch.zeros((2, 28), dtype=torch.int8)
        preorder = torch.tensor([[0, 5], [6, 4], [12, 12], [10, 10]], dtype=torch.int8).T
        preorder = torch.cat([preorder, zeros], dim=1)
        for i in range(32):
            tree.add(preorder.T[i], i)

        print(tree.equation_string())
        print(tree.rules)

    def test_simplify_equation(self):
        equation = "((c[0]-c[1])*(torch.sin((((torch.tensor(1, device=device)**torch.tensor(1, device=device))*(((x[0]*x[1])+c[2])/(c[3]+torch.tensor(1, device=device))))/(((torch.tensor(1, device=device)+torch.tensor(1, device=device))+(x[1]+torch.tensor(1, device=device)))+(c[4]+(c[5]**x[0])))))))"
        equ = simplify_equation(equation, 6)
        print(equ)


if __name__ == '__main__':
    unittest.main()
