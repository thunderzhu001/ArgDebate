import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os

class QBAFVisualizer:
    """
    Visualizes Quantitative Bipolar Argumentation Frameworks (QBAFs).
    
    Nodes represent arguments, with colors indicating final strengths.
    Edges represent Attack (red) and Support (green) relations.
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize(self, 
                  arg_ids: List[str], 
                  strengths: Dict[str, float], 
                  attacks: List[Tuple[str, str]], 
                  supports: List[Tuple[str, str]], 
                  filename: str = "qbaf_graph.png"):
        """
        Generates a visualization of the QBAF.
        
        Args:
            arg_ids (List[str]): List of argument IDs.
            strengths (Dict[str, float]): Final strengths of arguments.
            attacks (List[Tuple[str, str]]): List of (attacker, target) tuples.
            supports (List[Tuple[str, str]]): List of (supporter, target) tuples.
            filename (str): Output filename.
        """
        G = nx.DiGraph()
        
        # Add nodes
        for aid in arg_ids:
            G.add_node(aid, strength=strengths.get(aid, 0.5))
            
        # Add edges
        for att, target in attacks:
            G.add_edge(att, target, type='attack')
        for sup, target in supports:
            G.add_edge(sup, target, type='support')
            
        # Layout
        pos = nx.spring_layout(G)
        
        # Node colors based on strength (blue gradient)
        node_colors = [strengths.get(n, 0.5) for n in G.nodes()]
        
        plt.figure(figsize=(10, 8))
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(G, pos, 
                                       node_color=node_colors, 
                                       cmap=plt.cm.Blues, 
                                       node_size=1500, 
                                       alpha=0.8)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # Draw edges
        attack_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'attack']
        support_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'support']
        
        nx.draw_networkx_edges(G, pos, edgelist=attack_edges, 
                               edge_color='red', arrowstyle='-|>', arrowsize=20, width=2)
        nx.draw_networkx_edges(G, pos, edgelist=support_edges, 
                               edge_color='green', arrowstyle='-|>', arrowsize=20, width=2)
        
        plt.title("ArgDebate: QBAF Argumentation Graph")
        plt.colorbar(nodes, label='Argument Strength')
        plt.axis('off')
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {output_path}")
        return output_path

if __name__ == "__main__":
    # Quick test
    viz = QBAFVisualizer()
    test_args = ["A", "B", "C"]
    test_strengths = {"A": 0.9, "B": 0.4, "C": 0.7}
    test_attacks = [("A", "B")]
    test_supports = [("C", "A")]
    viz.visualize(test_args, test_strengths, test_attacks, test_supports, "test_viz.png")
