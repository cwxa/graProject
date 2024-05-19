import os
import re
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


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


def process_code_folder(folder_path, language, model):
    """
    Process a folder containing code files.

    Args:
        folder_path (str): Path to the folder.
        language (str): The programming language.
        model (nn.Module): The GCN model.

    Returns:
        tuple: Semantic complexity, compactness, degree centrality, clustering coefficient,
               and betweenness centrality.
    """
    code_processor = CodeProcessor()
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_path.endswith(f'.{language}'):  # Ensure processing only the correct language files
            code_processor.process_code_file(file_path)

    graph_data = create_graph_dataset(code_processor.graph, input_dim=10)
    output = model(graph_data.x, graph_data.edge_index)
    semantic_complexity = torch.mean(output).item()
    compactness = torch.std(output).item()

    degree_centrality = nx.degree_centrality(code_processor.graph)
    clustering_coefficient = nx.average_clustering(code_processor.graph)
    betweenness_centrality = nx.betweenness_centrality(code_processor.graph)

    print_analysis_results(language, Semantic_complexity=semantic_complexity, Compactness=compactness,
                           Degree_centrality=degree_centrality, Clustering_coefficient=clustering_coefficient,
                           Betweenness_centrality=betweenness_centrality)
    return semantic_complexity, compactness, degree_centrality, clustering_coefficient, betweenness_centrality


def compare_languages(go_results, c_results):
    """
    Compare the analysis results of Go and C languages.

    Args:
        go_results (tuple): Results of Go language analysis.
        c_results (tuple): Results of C language analysis.
    """
    go_semantic_complexity, go_compactness, go_degree_centrality, go_clustering_coefficient, go_betweenness_centrality = go_results
    c_semantic_complexity, c_compactness, c_degree_centrality, c_clustering_coefficient, c_betweenness_centrality = c_results

    comparisons = {
        "Semantic Complexity": "Go" if go_semantic_complexity > c_semantic_complexity else "C",
        "Compactness": "Go" if go_compactness > c_compactness else "C",
        "Degree Centrality": "Go" if sum(go_degree_centrality.values()) > sum(c_degree_centrality.values()) else "C",
        "Clustering Coefficient": "Go" if go_clustering_coefficient > c_clustering_coefficient else "C",
        "Betweenness Centrality": "Go" if sum(go_betweenness_centrality.values()) > sum(
            c_betweenness_centrality.values()) else "C"
    }

    print("Comparison Analysis:")
    for metric, winner in comparisons.items():
        print(f"{metric}: {winner} language code has higher {metric.lower()}")


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


def process_and_analyze_code(language, folder_path, model):
    """
    Process code files and perform analysis for a specific programming language.

    Args:
        language (str): The programming language.
        folder_path (str): Path to the folder containing code files.
        model (nn.Module): The GCN model.

    Returns:
        tuple: Results of analysis (semantic complexity, compactness, degree centrality, clustering coefficient, betweenness centrality).
    """
    results = process_code_folder(folder_path, language, model)
    print_analysis_results(language, Semantic_complexity=results[0], Compactness=results[1],
                           Degree_centrality=results[2], Clustering_coefficient=results[3],
                           Betweenness_centrality=results[4])
    return results


def main():
    # Define configurable parameters for GCN model
    input_dim = 10  # Adjusted input dimension
    hidden_dim = 64
    output_dim = 1

    # Initialize GCN model
    model = GCN(input_dim, hidden_dim, output_dim)

    # Define the languages and their respective folder paths
    languages = {
        "go": "gosrc",
        "c": "csrc"
    }

    # Ensure the output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for language, folder_path in languages.items():
        print(f"\nProcessing {language.upper()} files...\n")

        # Define the output subdirectory for the current language
        language_output_dir = os.path.join(output_dir, language)
        if not os.path.exists(language_output_dir):
            os.makedirs(language_output_dir)

        # Initialize CodeProcessor for each language
        code_processor = CodeProcessor()

        # Initialize tqdm for progress bar
        file_list = [f for f in os.listdir(folder_path) if f.endswith(f'.{language}')]
        total_files = len(file_list)
        with tqdm(total=total_files, desc=f"Processing {language} files", unit='file') as pbar:
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                code_processor.process_code_file(file_path)
                pbar.update(1)

        # Create graph dataset
        graph_data = create_graph_dataset(code_processor.graph, input_dim=input_dim)

        # Perform analysis using the GCN model
        output = model(graph_data.x, graph_data.edge_index)
        semantic_complexity = torch.mean(output).item()
        compactness = torch.std(output).item()

        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(code_processor.graph)
        clustering_coefficient = nx.average_clustering(code_processor.graph)
        betweenness_centrality = nx.betweenness_centrality(code_processor.graph)

        # Print analysis results
        print_analysis_results(language, Semantic_complexity=semantic_complexity, Compactness=compactness,
                               Degree_centrality=degree_centrality, Clustering_coefficient=clustering_coefficient,
                               Betweenness_centrality=betweenness_centrality)

        # Visualize and save graph for each file with progress bar
        with tqdm(total=total_files, desc=f"Visualizing {language} graphs", unit='graph') as pbar:
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                # Define the path where the graph visualization will be saved
                save_path = os.path.join(language_output_dir, f"{file_name}_structure.png")
                # Reset the code processor to process only the current file
                code_processor = CodeProcessor()
                code_processor.process_code_file(file_path)
                # Visualize the graph and save the image
                visualize_graph(code_processor.graph, save_path=save_path)
                pbar.update(1)

if __name__ == "__main__":
    main()
