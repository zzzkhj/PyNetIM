from typing import Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import IMGraphCpp

class IndependentCascadeModel:
    def __init__(self, graph: 'IMGraphCpp', seeds: Set[int]) -> None: ...
    
    def set_seeds(self, new_seeds: Set[int]) -> None: ...
    
    def run_monte_carlo_diffusion(self, rounds: int, seed: int = 0, use_multithread: bool = False, num_threads: int = 0) -> float: ...
