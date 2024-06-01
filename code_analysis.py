import os
import ast
import clang.cindex
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from subprocess import check_output
import json

class CodeProcessor:
    def __init__(self):
        self.graph = nx.DiGraph()

    def process_code_file(self, file_path):
        if file_path.endswith('.go'):
            self._process_go_file(file_path)
        elif file_path.endswith('.c'):
            self._process_c_file(file_path)

    def _process_go_file(self, file_path):
        try:
            output = check_output(['go', 'list', '-json', file_path])
            data = json.loads(output)
            self._traverse_go_ast(data)
        except Exception as e:
            print(f"Error processing Go file {file_path}: {e}")

    def _traverse_go_ast(self, data):
        if 'Name' in data:
            func_name = data['Name']
            self.graph.add_node(func_name)
            if 'Calls' in data:
                for call in data['Calls']:
                    self.graph.add_edge(func_name, call)

    def _process_c_file(self, file_path):
        index = clang.cindex.Index.create()
        translation_unit = index.parse(file_path)
        self._traverse_clang_ast(translation_unit.cursor)

    def _traverse_clang_ast(self, cursor):
        if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            self.graph.add_node(cursor.spelling)
            for child in cursor.get_children():
                if child.kind == clang.cindex.CursorKind.CALL_EXPR:
                    self.graph.add_edge(cursor.spelling, child.spelling)
        for child in cursor.get_children():
            self._traverse_clang_ast(child)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x, edge_index):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x)
        return x

def create_graph_dataset(graph, input_dim):
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    x = torch.eye(len(nodes), input_dim)
    edge_index = torch.tensor([[nodes.index(src), nodes.index(dst)] for src, dst in edges], dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)
