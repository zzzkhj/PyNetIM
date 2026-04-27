"""
Comprehensive Algorithm Comparison: Influence Spread Analysis

This script compares all influence maximization algorithms in PyNetIM.

================================================================================
EXPERIMENT SETTINGS
================================================================================

Network Configuration:
    - Type: Erdős-Rényi (ER) random graph
    - Nodes: 200
    - Edge probability: 0.02
    - Directed: Yes (bidirectional edges from undirected base)
    - Edge weights: WC (Weighted Cascade) model
      - p(u,v) = 1 / in_degree(v)
      - This is the standard WC model used in IM research

Seed Selection:
    - Seed size k: 10
    - MC rounds for evaluation: 1000

Algorithms Compared (25 total):
    1. Simulation-based: Greedy, CELF, CELF++
    2. RIS-based: IMM, TIM, TIM+, OPIM-C
    3. Heuristic: Degree, PageRank, VoteRank, KShell, Betweenness, 
                  Closeness, EigenVector, SingleDiscount, DegreeDiscount
    4. Reinforcement Learning: CoreQ, TCQ
    5. Population-based: RLSetGWO, SADPEA
    6. Deep Learning (pretrained): BiGDN, BiGDNS, ToupleGDD, S2VDQN
    7. Baseline: Random

Hardware Requirements:
    - CPU: Any modern processor
    - GPU: Not required (deep learning algorithms run on CPU)
    - RAM: 4GB minimum

Software Requirements:
    - Python >= 3.8
    - PyNetIM and dependencies
    - matplotlib, networkx, numpy

Output:
    - Console: Algorithm results and ranking table
    - Figures: Network visualization, influence comparison, spread distribution

Usage:
    python tests/test_algorithm_comparison.py

Note:
    - Deep learning algorithms require pretrained weights
    - Greedy/CELF algorithms may take several seconds
    - Total runtime: ~15-20 seconds
================================================================================
"""

import random
import time
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from pynetim import IMGraph
from pynetim.graph import generate_er_graph, set_wc_weights
from pynetim.algorithms import (
    DegreeCentralityAlgorithm, PageRankAlgorithm, VoteRankAlgorithm,
    KShellDecompositionAlgorithm, BetweennessCentralityAlgorithm,
    ClosenessCentralityAlgorithm, EigenvectorCentralityAlgorithm,
    SingleDiscountAlgorithm, DegreeDiscountAlgorithm,
    GreedyAlgorithm, CELFAlgorithm, CELFPlusAlgorithm,
    IMMAlgorithm, TIMAlgorithm, TIMPlusAlgorithm, OPIMCAlgorithm,
    CoreQAlgorithm, TCQAlgorithm,
    RLSetGWOAlgorithm, SADPEAAlgorithm,
    BiGDNAlgorithm, BiGDNSAlgorithm, ToupleGDDAlgorithm, S2VDQNAlgorithm,
)
from pynetim.diffusion_model import IndependentCascadeModel


def create_wc_network(n: int = 200, p: float = 0.02, seed: int = 42) -> IMGraph:
    """Create a network with WC (Weighted Cascade) edge weights.
    
    WC model: p(u,v) = 1 / in_degree(v)
    This is the standard WC model used in IM research.
    
    Uses ER random graph for more uniform degree distribution.
    
    Args:
        n: Number of nodes.
        p: Edge probability.
        seed: Random seed for reproducibility.
        
    Returns:
        IMGraph: Graph with WC edge weights.
    """
    print(f"Creating ER random network with WC weights (n={n}, p={p})...")
    
    # Generate undirected ER graph using pynetim built-in function
    graph = generate_er_graph(n=n, p=p, directed=False, random_seed=seed)
    
    # Convert to bidirectional directed graph
    edges = []
    for u in range(graph.num_nodes):
        for v in graph.out_neighbors(u):
            edges.append((u, v))
            edges.append((v, u))
    
    graph = IMGraph(edges, directed=True)
    
    # Set WC weights using pynetim built-in function
    set_wc_weights(graph)
    
    # Collect weights for statistics
    weights = []
    for u in range(graph.num_nodes):
        for v in graph.out_neighbors(u):
            weights.append(graph.get_edge_weight(u, v))
    
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.num_edges}")
    print(f"  Avg degree: {sum(graph.degree(i) for i in range(graph.num_nodes)) / graph.num_nodes:.2f}")
    print(f"  Weight range: [{min(weights):.4f}, {max(weights):.4f}]")
    print(f"  Weight mean: {np.mean(weights):.4f}")
    
    return graph


def run_algorithm(name: str, algo_class, graph: IMGraph, k: int, mc_rounds: int = 500, **kwargs) -> Tuple[Set[int], float]:
    """Run an algorithm and return seeds and runtime.
    
    Args:
        name: Algorithm name.
        algo_class: Algorithm class.
        graph: Input graph.
        k: Seed size.
        mc_rounds: MC rounds for simulation-based algorithms.
        **kwargs: Additional arguments for the algorithm.
        
    Returns:
        Tuple[Set[int], float]: Seeds and runtime.
    """
    start_time = time.time()
    
    algo = algo_class(graph, **kwargs)
    
    if name in ['Greedy', 'CELF', 'CELF++']:
        seeds = algo.run(k=k, mc_rounds=mc_rounds, show_progress=False)
    elif name == 'OPIM-C':
        seeds = algo.run(k=k, epsilon=0.1)
    else:
        seeds = algo.run(k=k)
    
    runtime = time.time() - start_time
    return seeds, runtime


def evaluate_influence(graph: IMGraph, seeds: Set[int], mc_rounds: int = 1000) -> float:
    """Evaluate influence spread using MC simulation.
    
    Args:
        graph: Input graph.
        seeds: Seed set.
        mc_rounds: Number of MC rounds.
        
    Returns:
        float: Mean influence spread.
    """
    model = IndependentCascadeModel(graph, seeds)
    return model.run_monte_carlo_diffusion(mc_rounds=mc_rounds)


def visualize_network(graph: IMGraph, seeds_dict: Dict[str, Set[int]], output_path: str, top_n: int = 6):
    """Visualize network with highlighted seed nodes.
    
    Args:
        graph: Input graph.
        seeds_dict: Dictionary of algorithm names to seed sets.
        output_path: Output file path.
        top_n: Number of top algorithms to visualize.
    """
    n_plots = min(top_n + 1, len(seeds_dict) + 1)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    nx_graph = nx.DiGraph()
    for u in range(graph.num_nodes):
        nx_graph.add_node(u)
    for u in range(graph.num_nodes):
        for v in graph.out_neighbors(u):
            nx_graph.add_edge(u, v)
    
    pos = nx.spring_layout(nx_graph, seed=42, k=2/np.sqrt(graph.num_nodes))
    
    all_seeds = set()
    for seeds in list(seeds_dict.values())[:top_n]:
        all_seeds.update(seeds)
    
    ax = axes[0]
    node_colors = ['red' if node in all_seeds else 'lightblue' for node in nx_graph.nodes()]
    nx.draw_networkx_edges(nx_graph, pos, ax=ax, alpha=0.1, arrows=False)
    nx.draw_networkx_nodes(nx_graph, pos, ax=ax, node_color=node_colors, node_size=20, alpha=0.7)
    ax.set_title(f'Network Structure\n(Red = Selected seeds)', fontsize=11)
    ax.axis('off')
    
    colors = plt.cm.tab10.colors
    
    for idx, (name, seeds) in enumerate(list(seeds_dict.items())[:top_n]):
        ax = axes[idx + 1]
        
        node_colors = [colors[idx % 10] if node in seeds else 'lightgray' for node in nx_graph.nodes()]
        nx.draw_networkx_edges(nx_graph, pos, ax=ax, alpha=0.05, arrows=False)
        nx.draw_networkx_nodes(nx_graph, pos, ax=ax, node_color=node_colors, node_size=20, alpha=0.7)
        ax.set_title(f'{name}\n({len(seeds)} seeds)', fontsize=10)
        ax.axis('off')
    
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Network Visualization (n={graph.num_nodes}, WC model)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Network visualization saved: {output_path}")


def plot_influence_comparison(results: Dict[str, Dict], output_path: str):
    """Plot influence spread comparison.
    
    Args:
        results: Dictionary of algorithm results.
        output_path: Output file path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
    names = [x[0] for x in sorted_results]
    means = [x[1]['mean'] for x in sorted_results]
    runtimes = [x[1]['runtime'] for x in sorted_results]
    
    ax1 = axes[0]
    x = np.arange(len(names))
    bars = ax1.bar(x, means, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Algorithm', fontsize=12)
    ax1.set_ylabel('Influence Spread (MC)', fontsize=12)
    ax1.set_title('Influence Spread Comparison (WC Model)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax2 = axes[1]
    colors = ['green' if rt < 0.1 else 'orange' if rt < 1 else 'red' for rt in runtimes]
    bars2 = ax2.bar(x, runtimes, alpha=0.7, color=colors)
    ax2.set_xlabel('Algorithm', fontsize=12)
    ax2.set_ylabel('Runtime (seconds)', fontsize=12)
    ax2.set_title('Runtime Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    for bar, rt in zip(bars2, runtimes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{rt:.2f}s', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Influence comparison saved: {output_path}")


def plot_spread_distribution(graph: IMGraph, seeds_dict: Dict[str, Set[int]], 
                              mc_rounds: int = 500, output_path: str = None, top_n: int = 6):
    """Plot the distribution of influence spread.
    
    Args:
        graph: Input graph.
        seeds_dict: Dictionary of algorithm names to seed sets.
        mc_rounds: Number of MC rounds.
        output_path: Output file path.
        top_n: Number of top algorithms to plot.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    top_algos = list(seeds_dict.items())[:top_n]
    
    for name, seeds in top_algos:
        spread_list = []
        for _ in range(mc_rounds):
            model = IndependentCascadeModel(graph, seeds)
            spread = model.run_single_simulation()
            spread_list.append(spread)
        
        ax.hist(spread_list, bins=30, alpha=0.5, label=f'{name} (mean={np.mean(spread_list):.1f})')
    
    ax.set_xlabel('Influence Spread', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Influence Spread (WC Model)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Spread distribution saved: {output_path}")


def main():
    """Main function to run the algorithm comparison."""
    random.seed(42)
    np.random.seed(42)
    
    print("=" * 70)
    print("Comprehensive Algorithm Comparison (WC Model)")
    print("=" * 70)
    
    n = 200
    k = 10
    mc_rounds = 1000
    
    graph = create_wc_network(n=n, p=0.02, seed=42)
    
    print(f"\nSeed size k = {k}")
    print(f"MC rounds = {mc_rounds}")
    
    algorithms = {
        'Greedy': (GreedyAlgorithm, {'diffusion_model': 'IC'}),
        'CELF': (CELFAlgorithm, {'diffusion_model': 'IC'}),
        'CELF++': (CELFPlusAlgorithm, {'diffusion_model': 'IC'}),
        'IMM': (IMMAlgorithm, {'model': 'IC', 'epsilon': 0.1}),
        'TIM': (TIMAlgorithm, {'model': 'IC', 'epsilon': 0.1}),
        'TIM+': (TIMPlusAlgorithm, {'model': 'IC', 'epsilon': 0.1}),
        'OPIM-C': (OPIMCAlgorithm, {'model': 'IC'}),
        'Degree': (DegreeCentralityAlgorithm, {}),
        'PageRank': (PageRankAlgorithm, {}),
        'VoteRank': (VoteRankAlgorithm, {}),
        'KShell': (KShellDecompositionAlgorithm, {}),
        'Betweenness': (BetweennessCentralityAlgorithm, {}),
        'Closeness': (ClosenessCentralityAlgorithm, {}),
        'EigenVector': (EigenvectorCentralityAlgorithm, {}),
        'SingleDiscount': (SingleDiscountAlgorithm, {}),
        'DegreeDiscount': (DegreeDiscountAlgorithm, {}),
        'CoreQ': (CoreQAlgorithm, {'n_candidates': 50, 'episodes': 100, 'random_seed': 42}),
        'TCQ': (TCQAlgorithm, {'n_candidates': 50, 'episodes': 100, 'random_seed': 42}),
        'RLSetGWO': (RLSetGWOAlgorithm, {'pop_size': 20, 'max_iter': 30}),
        'SADPEA': (SADPEAAlgorithm, {'pop_size': 20, 'max_iter': 30, 'random_seed': 42}),
        'BiGDN': (BiGDNAlgorithm, {'pretrained': True, 'device': 'cpu'}),
        'BiGDNS': (BiGDNSAlgorithm, {'pretrained': True, 'device': 'cpu'}),
        'ToupleGDD': (ToupleGDDAlgorithm, {'pretrained': True, 'device': 'cpu'}),
        'S2VDQN': (S2VDQNAlgorithm, {'pretrained': True, 'device': 'cpu'}),
        'Random': (None, {}),
    }
    
    results = {}
    seeds_dict = {}
    
    print("\n" + "-" * 70)
    print("Running algorithms...")
    print("-" * 70)
    
    for name, (algo_class, kwargs) in algorithms.items():
        print(f"\n{name}:", end=" ")
        
        try:
            if algo_class is None:
                start_time = time.time()
                seeds = set(random.sample(range(graph.num_nodes), k))
                runtime = time.time() - start_time
            else:
                seeds, runtime = run_algorithm(name, algo_class, graph, k, mc_rounds=500, **kwargs)
            
            print(f"seeds={sorted(seeds)[:5]}..., time={runtime:.2f}s", end="")
            
            influence = evaluate_influence(graph, seeds, mc_rounds=mc_rounds)
            print(f", influence={influence:.2f}")
            
            results[name] = {
                'seeds': seeds,
                'mean': influence,
                'runtime': runtime,
            }
            seeds_dict[name] = seeds
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("Summary (Sorted by Influence)")
    print("=" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Algorithm':<15} {'Influence':<12} {'Runtime':<12}")
    print("-" * 50)
    
    for rank, (name, data) in enumerate(sorted_results, 1):
        print(f"{rank:<5} {name:<15} {data['mean']:.2f}       {data['runtime']:.2f}s")
    
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)
    
    sorted_seeds_dict = {name: results[name]['seeds'] for name, _ in sorted_results}
    
    output_dir = '/root/PyNetIM/tests'
    visualize_network(graph, sorted_seeds_dict, f'{output_dir}/fig_network_visualization.png', top_n=6)
    plot_influence_comparison(results, f'{output_dir}/fig_influence_comparison.png')
    plot_spread_distribution(graph, sorted_seeds_dict, mc_rounds=500, 
                             output_path=f'{output_dir}/fig_spread_distribution.png', top_n=6)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()
