import os
import re
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import random


class CodeProcessor:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.current_function = None

    def reset_graph(self):
        self.graph = nx.DiGraph()

    def process_code_file(self, file_path):
        self.reset_graph()
        code = self.read_code_file(file_path)
        if code is None:
            return None

        if file_path.endswith('.go'):
            self.process_go_code(code)
        elif file_path.endswith('.c'):
            self.process_c_code(code)

        return self.graph

    def read_code_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                code = file.readlines()  # 使用readlines逐行读取文件
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None
        except IOError as e:
            print(f"IO error while reading file '{file_path}': {e}")
            return None
        except Exception as e:
            print(f"Unexpected error reading file '{file_path}': {e}")
            return None
        return ''.join(code)  # 将列表转换为字符串

    def process_go_code(self, code):
        lines = code.split('\n')
        for line in lines:
            if line.startswith("func"):
                parts = line.split()
                if len(parts) > 1:
                    function_name = parts[1].split('(')[0]
                    self.add_function_node(function_name)

    def process_c_code(self, code):
        pattern = r'\b\w+\s+\w+\s*\([^)]*\)\s*{'
        functions = re.findall(pattern, code)
        for function in functions:
            function_name = function.split()[1].split('(')[0]
            self.add_function_node(function_name)

    def add_function_node(self, function_name):
        if function_name not in self.graph:
            self.graph.add_node(function_name)
        if self.current_function:
            self.graph.add_edge(self.current_function, function_name)
        self.current_function = function_name


def save_graph_visualization(graph, output_path):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold',
            arrows=True, linewidths=1, edge_color='gray', alpha=0.7)
    plt.title("Function Call Graph")
    plt.savefig(output_path)
    plt.close()


def save_total_graph_visualization(graph, output_path):
    plt.figure(figsize=(20, 16))  # Increase figure size for better visualization
    pos = nx.spring_layout(graph, seed=42, k=0.5)  # Adjust spring layout parameters for better spacing

    # Draw nodes
    node_colors = [plt.cm.tab20(random.randint(0, 19)) for _ in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_size=3000, node_color=node_colors, alpha=0.9)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, edge_color='gray', alpha=0.6, width=2)

    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight='bold')

    plt.title("Total Function Call Graph")
    plt.savefig(output_path)
    plt.close()


def process_code_folder(folder_path, language):
    code_processor = CodeProcessor()

    file_list = [f for f in os.listdir(folder_path) if f.endswith(f'.{language}')]
    total_files = len(file_list)
    output_dir = os.path.join("output", language)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_graphs = []
    with tqdm(total=total_files, desc=f"Processing {language} files", unit='file') as pbar:
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            graph = code_processor.process_code_file(file_path)
            if graph is not None:
                save_graph_visualization(graph, os.path.join(output_dir, f"{file_name}_graph.png"))
                all_graphs.append(graph)
            pbar.update(1)

    total_graph = merge_graphs(all_graphs)
    save_total_graph_visualization(total_graph, os.path.join(output_dir, f"total_graph_{language}.png"))

    return all_graphs


def merge_graphs(graph_list):
    merged_graph = nx.DiGraph()
    for graph in graph_list:
        merged_graph.add_edges_from(graph.edges())
    return merged_graph


def calculate_metrics(graph_list):
    semantic_complexity = []
    density = []
    degree_centrality = []
    clustering_coefficient = []
    betweenness_centrality = []

    for graph in graph_list:
        # 语义复杂性：节点数量
        semantic_complexity.append(len(graph.nodes()))

        # 紧密度
        density.append(nx.density(graph))

        # 度中心性
        degree_centrality.extend(list(nx.degree_centrality(graph).values()))

        # 聚类系数
        clustering_coefficient.extend(list(nx.clustering(graph).values()))

        # 介数中心性
        betweenness_centrality.extend(list(nx.betweenness_centrality(graph).values()))

    return semantic_complexity, density, degree_centrality, clustering_coefficient, betweenness_centrality


def plot_boxplots(data, labels, title, ylabel, output_path):
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Language")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def main():
    languages = {
        "go": "gosrc",
        "c": "csrc"
    }

    output_dir =    "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    semantic_complexity_data = []
    density_data = []
    degree_centrality_data = []
    clustering_coefficient_data = []
    betweenness_centrality_data = []
    language_labels = []

    for language, folder_path in languages.items():
        print(f"\nProcessing {language.upper()} files...\n")
        graphs = process_code_folder(folder_path, language)

        semantic_complexity, density, degree_centrality, clustering_coefficient, betweenness_centrality = calculate_metrics(
            graphs)

        semantic_complexity_data.append(semantic_complexity)
        density_data.append(density)
        degree_centrality_data.append(degree_centrality)
        clustering_coefficient_data.append(clustering_coefficient)
        betweenness_centrality_data.append(betweenness_centrality)
        language_labels.append(language.upper())

    plot_boxplots(semantic_complexity_data, language_labels, "Semantic Complexity", "Number of Nodes",
                  os.path.join(output_dir, "semantic_complexity_boxplot.png"))
    plot_boxplots(density_data, language_labels, "Density", "Density",
                  os.path.join(output_dir, "density_boxplot.png"))
    plot_boxplots(degree_centrality_data, language_labels, "Degree Centrality", "Degree Centrality",
                  os.path.join(output_dir, "degree_centrality_boxplot.png"))
    plot_boxplots(clustering_coefficient_data, language_labels, "Clustering Coefficient", "Clustering Coefficient",
                  os.path.join(output_dir, "clustering_coefficient_boxplot.png"))
    plot_boxplots(betweenness_centrality_data, language_labels, "Betweenness Centrality", "Betweenness Centrality",
                  os.path.join(output_dir, "betweenness_centrality_boxplot.png"))


if __name__ == "__main__":
    main()