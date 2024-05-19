import os
from tqdm import tqdm
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from code_analysis import CodeProcessor, GCN, create_graph_dataset

def process_code_file(code_processor, file_path, model):
    code_processor.process_code_file(file_path)

    # Check if the graph has at least one edge
    if len(code_processor.graph.edges) == 0:
        print(f"Skipping {file_path} as it has no edges.")
        return None, None, None, None, None

    graph_data = create_graph_dataset(code_processor.graph, input_dim=10)
    output = model(graph_data.x, graph_data.edge_index)
    semantic_complexity = torch.mean(output).item()
    compactness = torch.std(output).item()

    degree_centrality = nx.degree_centrality(code_processor.graph)
    clustering_coefficient = nx.average_clustering(code_processor.graph)
    betweenness_centrality = nx.betweenness_centrality(code_processor.graph)

    return semantic_complexity, compactness, np.mean(list(degree_centrality.values())), clustering_coefficient, np.mean(list(betweenness_centrality.values()))

def process_code_folder(folder_path, language, model):
    code_processor = CodeProcessor()
    all_semantic_complexity = []
    all_compactness = []
    all_degree_centrality = []
    all_clustering_coefficient = []
    all_betweenness_centrality = []

    file_list = [f for f in os.listdir(folder_path) if f.endswith(f'.{language}')]
    total_files = len(file_list)
    with tqdm(total=total_files, desc=f"Processing {language} files", unit='file') as pbar:
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            result = process_code_file(code_processor, file_path, model)
            if result:
                semantic_complexity, compactness, degree_centrality, clustering_coefficient, betweenness_centrality = result
                all_semantic_complexity.append(semantic_complexity)
                all_compactness.append(compactness)
                all_degree_centrality.append(degree_centrality)
                all_clustering_coefficient.append(clustering_coefficient)
                all_betweenness_centrality.append(betweenness_centrality)
            pbar.update(1)

    return all_semantic_complexity, all_compactness, all_degree_centrality, all_clustering_coefficient, all_betweenness_centrality

def save_boxplot(data, labels, metric, languages, output_dir):
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=labels)
    plt.title(f'{metric} Comparison between Go and C')
    plt.xlabel('Language')
    plt.ylabel(metric)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric.lower().replace(" ", "_")}_comparison.png'))
    plt.close()

def main():
    input_dim = 10
    hidden_dim = 64
    output_dim = 1
    model = GCN(input_dim, hidden_dim, output_dim)

    languages = {
        "go": "gosrc",
        "c": "csrc"
    }

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    language_results = {}

    for language, folder_path in languages.items():
        print(f"\nProcessing {language.upper()} files...\n")
        language_output_dir = os.path.join(output_dir, language)
        if not os.path.exists(language_output_dir):
            os.makedirs(language_output_dir)
        language_results[language] = process_code_folder(folder_path, language, model)

    # Save boxplots for each metric and language
    metrics = ["Semantic Complexity", "Compactness", "Degree Centrality", "Clustering Coefficient", "Betweenness Centrality"]
    for metric in metrics:
        data = [language_results[lang][metrics.index(metric)] for lang in languages.keys()]
        save_boxplot(data, languages.keys(), metric, languages, output_dir)

if __name__ == "__main__":
    main()
