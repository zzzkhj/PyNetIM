import psutil
import os
import time
from typing import Dict, Any, Optional

class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self.get_memory_info()['rss']
        self.peak_memory = self.baseline_memory
    
    def get_memory_info(self) -> Dict[str, Any]:
        mem_info = self.process.memory_info()
        return {
            'rss': mem_info.rss / 1024 / 1024,
            'vms': mem_info.vms / 1024 / 1024,
            'shared': mem_info.shared / 1024 / 1024,
            'text': mem_info.text / 1024 / 1024,
            'lib': mem_info.lib / 1024 / 1024,
            'data': mem_info.data / 1024 / 1024,
            'dirty': mem_info.dirty / 1024 / 1024
        }
    
    def get_memory_usage(self) -> float:
        mem_info = self.get_memory_info()
        current_rss = mem_info['rss']
        self.peak_memory = max(self.peak_memory, current_rss)
        return current_rss
    
    def get_memory_increase(self) -> float:
        current_memory = self.get_memory_usage()
        return current_memory - self.baseline_memory
    
    def get_peak_memory(self) -> float:
        return self.peak_memory
    
    def reset_baseline(self):
        self.baseline_memory = self.get_memory_usage()
        self.peak_memory = self.baseline_memory
    
    def __enter__(self):
        self.reset_baseline()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def profile_memory_usage(func, *args, **kwargs) -> Dict[str, Any]:
    monitor = MemoryMonitor()
    start_time = time.time()
    
    result = func(*args, **kwargs)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    memory_info = monitor.get_memory_info()
    memory_increase = monitor.get_memory_increase()
    peak_memory = monitor.get_peak_memory()
    
    return {
        'result': result,
        'elapsed_time': elapsed_time,
        'memory_info': memory_info,
        'memory_increase_mb': memory_increase,
        'peak_memory_mb': peak_memory
    }

def format_memory_size(size_mb: float) -> str:
    if size_mb < 1024:
        return f"{size_mb:.2f} MB"
    else:
        return f"{size_mb / 1024:.2f} GB"

def print_memory_stats(stats: Dict[str, Any], label: str = ""):
    if label:
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")
    
    print(f"Elapsed time: {stats['elapsed_time']:.4f} seconds")
    print(f"Memory increase: {format_memory_size(stats['memory_increase_mb'])}")
    print(f"Peak memory: {format_memory_size(stats['peak_memory_mb'])}")
    print(f"\nDetailed memory info:")
    for key, value in stats['memory_info'].items():
        print(f"  {key}: {format_memory_size(value)}")

if __name__ == "__main__":
    print("Memory Monitor Test")
    print("=" * 60)
    
    monitor = MemoryMonitor()
    print(f"Baseline memory: {format_memory_size(monitor.get_memory_usage())}")
    
    data = [i for i in range(1000000)]
    print(f"After creating list: {format_memory_size(monitor.get_memory_usage())}")
    print(f"Memory increase: {format_memory_size(monitor.get_memory_increase())}")
    
    del data
    print(f"After deleting list: {format_memory_size(monitor.get_memory_usage())}")
    
    print("\nTesting with context manager:")
    with MemoryMonitor() as m:
        large_data = [i for i in range(5000000)]
        time.sleep(0.1)
        print(f"Peak memory: {format_memory_size(m.get_peak_memory())}")
        print(f"Memory increase: {format_memory_size(m.get_memory_increase())}")
