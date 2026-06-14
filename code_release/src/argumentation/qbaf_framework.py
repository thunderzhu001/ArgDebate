"""
Quadruple-Based Argumentation Framework (QBAF) implementation.
Supports cyclic attack and support relations with DF-QuAD semantics.
"""

from typing import List, Dict, Tuple, Set, Optional
import numpy as np


class QBAFramework:
    """
    A Quadruple-Based Argumentation Framework (QBAF) that supports:
    - Arguments with initial strengths
    - Attack relations
    - Support relations
    - Cyclic dependencies (non-DAG)
    - DF-QuAD semantics for iterative evaluation
    """
    
    def __init__(self, arguments: List[str], initial_strengths: List[float],
                 attacks: List[Tuple[str, str]], supports: List[Tuple[str, str]],
                 semantics: str = "DFQuAD_model",
                 attack_weights: Optional[Dict[Tuple[str, str], float]] = None,
                 support_weights: Optional[Dict[Tuple[str, str], float]] = None,
                 quality_scores: Optional[Dict[str, float]] = None):
        """
        Initialize QBAF.

        Args:
            arguments: List of argument IDs
            initial_strengths: Initial strength (0-1) for each argument
            attacks: List of (attacker_id, attacked_id) tuples
            supports: List of (supporter_id, supported_id) tuples
            semantics: Evaluation semantics (currently only "DFQuAD_model")
            quality_scores: Optional quality scores (0-1) for each argument
        """
        self.arguments = set(arguments)
        self.semantics = semantics

        # Store initial strengths
        self.initial_strengths_dict = {arg: strength for arg, strength in zip(arguments, initial_strengths)}

        # Store quality scores (default to 1.0 if not provided)
        self.quality_scores = quality_scores or {arg: 1.0 for arg in arguments}

        # Store relations as sets for efficient lookup
        self.attacks = set(attacks)
        self.supports = set(supports)
        self.attack_weights = {
            edge: float(max(0.0, min(1.0, w))) for edge, w in (attack_weights or {}).items()
        }
        self.support_weights = {
            edge: float(max(0.0, min(1.0, w))) for edge, w in (support_weights or {}).items()
        }
        
        # Build reverse indices for efficient querying
        self._build_indices()
    
    def _build_indices(self):
        """Build reverse indices for efficient querying."""
        self._attackers = {arg: set() for arg in self.arguments}
        self._supporters = {arg: set() for arg in self.arguments}
        self._attacked = {arg: set() for arg in self.arguments}
        self._supported = {arg: set() for arg in self.arguments}
        
        for attacker, attacked in self.attacks:
            self._attackers[attacked].add(attacker)
            self._attacked[attacker].add(attacked)
        
        for supporter, supported in self.supports:
            self._supporters[supported].add(supporter)
            self._supported[supporter].add(supported)
    
    def initial_strength(self, arg_id: str) -> float:
        """Get initial strength of an argument."""
        return self.initial_strengths_dict.get(arg_id, 0.5)
    
    def attackersOf(self, arg_id: str) -> List[str]:
        """Get all arguments that attack the given argument."""
        return list(self._attackers.get(arg_id, set()))
    
    def supportersOf(self, arg_id: str) -> List[str]:
        """Get all arguments that support the given argument."""
        return list(self._supporters.get(arg_id, set()))
    
    def attackedBy(self, arg_id: str) -> List[str]:
        """Get all arguments attacked by the given argument."""
        return list(self._attacked.get(arg_id, set()))
    
    def supportedBy(self, arg_id: str) -> List[str]:
        """Get all arguments supported by the given argument."""
        return list(self._supported.get(arg_id, set()))

    def attack_weight(self, attacker_id: str, attacked_id: str) -> float:
        return float(self.attack_weights.get((attacker_id, attacked_id), 1.0))

    def support_weight(self, supporter_id: str, supported_id: str) -> float:
        return float(self.support_weights.get((supporter_id, supported_id), 1.0))
    
    def evaluate(self, max_iter: int = 100, tolerance: float = 1e-4) -> Dict[str, float]:
        """
        Evaluate the framework using DF-QuAD semantics with iterative fixed-point computation.
        Supports cyclic relations.
        
        Args:
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary mapping argument IDs to their final strengths (0-1)
        """
        arg_ids = list(self.arguments)
        n = len(arg_ids)
        
        # Initialize with base strengths
        base_scores = np.array([self.initial_strength(aid) for aid in arg_ids])
        current_strengths = base_scores.copy().astype(float)
        
        # Create ID to index mapping
        id_to_idx = {aid: idx for idx, aid in enumerate(arg_ids)}
        
        for iteration in range(max_iter):
            prev_strengths = current_strengths.copy()
            new_strengths = np.zeros(n)
            
            for idx, aid in enumerate(arg_ids):
                # Get attackers and supporters
                attackers = self.attackersOf(aid)
                supporters = self.supportersOf(aid)
                
                # Get weighted strengths (with quality scores)
                att_strengths = [
                    current_strengths[id_to_idx[att]] * self.attack_weight(att, aid) * self.quality_scores.get(att, 1.0)
                    for att in attackers
                ]
                sup_strengths = [
                    current_strengths[id_to_idx[sup]] * self.support_weight(sup, aid) * self.quality_scores.get(sup, 1.0)
                    for sup in supporters
                ]
                
                # Aggregate using product-based approach
                def aggregate(scores):
                    if not scores:
                        return 0.0
                    result = 0.0
                    for score in scores:
                        result = result + score - result * score
                    return min(result, 0.999)  # Avoid numerical issues
                
                inh = aggregate(att_strengths)  # Inhibition from attackers
                exc = aggregate(sup_strengths)  # Excitation from supporters
                
                # DF-QuAD update formula
                base = base_scores[idx]
                
                # Handle edge cases to avoid division by zero
                inh = min(inh, 1.0 - 1e-10)
                exc = min(exc, 1.0 - 1e-10)
                
                if exc >= inh:
                    if inh < 1.0:
                        strength = base + (1 - base) * ((exc - inh) / (1 - inh))
                    else:
                        strength = base + (1 - base) * exc
                else:
                    if exc < 1.0:
                        strength = base * (1 - ((inh - exc) / (1 - exc)))
                    else:
                        strength = base * (1 - inh)
                
                # Ensure strength stays in [0, 1]
                new_strengths[idx] = max(0.0, min(1.0, strength))
            
            current_strengths = new_strengths
            
            # Check convergence
            diff = np.linalg.norm(current_strengths - prev_strengths)
            if diff < tolerance:
                break
        
        return {aid: float(current_strengths[id_to_idx[aid]]) for aid in arg_ids}


def visualize_qbaf(framework: QBAFramework, output_path: str = "qbaf_visualization.png"):
    """
    Visualize the QBAF using NetworkX and Matplotlib.
    
    Args:
        framework: QBAFramework instance
        output_path: Path to save the visualization
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: NetworkX or Matplotlib not available for visualization")
        return
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for arg in framework.arguments:
        G.add_node(arg)
    
    # Add edges for attacks (red) and supports (green)
    for attacker, attacked in framework.attacks:
        G.add_edge(attacker, attacked, relation='attack')
    
    for supporter, supported in framework.supports:
        G.add_edge(supporter, supported, relation='support')
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw attack edges in red
    attack_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'attack']
    nx.draw_networkx_edges(G, pos, edgelist=attack_edges, edge_color='red', 
                          arrows=True, arrowsize=20, width=2, label='Attack')
    
    # Draw support edges in green
    support_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'support']
    nx.draw_networkx_edges(G, pos, edgelist=support_edges, edge_color='green',
                          arrows=True, arrowsize=20, width=2, label='Support')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Quadruple-Based Argumentation Framework")
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    # Visualization saved silently
    plt.close()
