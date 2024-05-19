import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt


class CodeProcessor:
    """
    Class to process code files and extract function relationships to build a graph.
    """

    def __init__(self):
        """
        Initialize CodeProcessor object.
        """
        self.graph = nx.DiGraph()
        self.current_function = None

    def process_code_file(self, file_path):
        """
        Process a code file and extract function relationships.

        Args:
            file_path (str): Path to the code file.
        """
        code = self.read_code_file(file_path)
        if code is None:
            return

        if file_path.endswith('.go'):
            self.process_go_code(code)
        elif file_path.endswith('.c'):
            self.process_c_code(code)
        self.current_function = None  # Reset current function after processing each file

    def read_code_file(self, file_path):
        """
        Read code from a file.

        Args:
            file_path (str): Path to the code file.

        Returns:
            str: The code read from the file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                code = file.read()
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")
            return None
        return code

    def process_go_code(self, code):
        """
        Process Go code and extract function relationships.

        Args:
            code (str): The Go code to process.
        """
        lines = code.split('\n')

        for line in lines:
            if line.startswith("func"):
                parts = line.split()
                if len(parts) > 1:  # Ensure at least two elements
                    function_name = parts[1].split('(')[0]
                    self.add_function_node(function_name)

    def process_c_code(self, code):
        """
        Process C code and extract function relationships.

        Args:
            code (str): The C code to process.
        """
        pattern = r'\b\w+\s+\w+\s*\([^)]*\)\s*{'
        functions = re.findall(pattern, code)

        for function in functions:
            function_name = function.split()[1].split('(')[0]
            self.add_function_node(function_name)

    def add_function_node(self, function_name):
        """
        Add a function node to the graph.

        Args:
            function_name (str): The name of the function.
        """
        self.graph.add_node(function_name)
        if self.current_function:
            self.graph.add_edge(self.current_function, function_name)
        self.current_function = function_name


def create_graph_dataset(graph, input_dim=10):
    """
    Create a PyTorch Geometric dataset from a graph.

    Args:
        graph (nx.Graph): The graph.
        input_dim (int): Input feature dimension.

    Returns:
        Data: PyTorch Geometric Data object.
    """
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    edge_index = torch.tensor([[mapping[edge[0]], mapping[edge[1]]] for edge in graph.edges()], dtype=torch.long).t().contiguous()
    num_nodes = len(graph.nodes())
    x = torch.randn((num_nodes, input_dim), dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


class GCN(nn.Module):
    """
    Graph Convolutional Network model.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize GCN model.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN model.

        Args:
            x (Tensor): Input features.
            edge_index (LongTensor): Graph edge indices.

        Returns:
            Tensor: Output tensor.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def print_analysis_results(language, **kwargs):
    """
    Print analysis results.

    Args:
        language (str): The programming language.
        kwargs (dict): Analysis results.
    """
    print(f"{language.capitalize()} Language Analysis Results:")
    for key, value in kwargs.items():
        if isinstance(value, dict):
            print(f"{key.capitalize()}:")
            for node, centrality_value in value.items():
                print(f"  {node}: {centrality_value}")
        else:
            print(f"{key.capitalize()}: {value}")
    print()


def visualize_graph(graph, node_color='lightblue', node_size=1000, edge_color='gray', linewidths=0.5, figsize=(10, 6),
                    save_path=None):
    """
    Visualize a graph.

    Args:
        graph (nx.Graph): The graph to visualize.
        node_color (str or list): Color(s) for the nodes.
        node_size (int or list): Size(s) for the nodes.
        edge_color (str or list): Color(s) for the edges.
        linewidths (float or list): Width(s) for the edges.
        figsize (tuple): Size of the plot (width, height).
        save_path (str): Path to save the image (optional).
    """
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color=node_color, node_size=node_size, edge_color=edge_color,
            linewidths=linewidths)
    plt.title("Code Structure Visualization")
    plt.axis('off')  # Turn off the axis
    plt.tight_layout()  # Adjust the layout automatically
    if save_path:
        plt.savefig(save_path)
        plt.close()  # Close the plot after saving
    else:
        plt.show()
