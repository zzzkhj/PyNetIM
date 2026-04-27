# -*- coding: utf-8 -*-
"""预训练权重管理器。

类似 Hugging Face Transformers 的权重管理机制，支持：
- 从本地缓存加载权重
- 从远程 URL 下载权重
- 版本管理和校验
"""

import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
import urllib.request
import warnings
from tqdm import tqdm

WEIGHTS_CACHE_DIR = Path.home() / ".pynetim" / "weights"
WEIGHTS_BASE_URL = "https://pynetim.yinjiy.cn/weights"

WEIGHTS_CONFIG: Dict[str, Dict[str, Any]] = {
    "bigdn_weights.pth": {
        "algorithm": "BiGDN",
        "description": "BiGDN teacher model weights",
        "filename": "bigdn_weights.pth",
        "url": f"{WEIGHTS_BASE_URL}/bigdn_weights.pth",
        "md5": None,
        "size_mb": 0.5,
    },
    "bigdns_weights.pth": {
        "algorithm": "BiGDNS",
        "description": "BiGDN student model weights (distilled)",
        "filename": "bigdns_weights.pth",
        "url": f"{WEIGHTS_BASE_URL}/bigdns_weights.pth",
        "md5": None,
        "size_mb": 0.3,
    },
    "node_encoder.pth": {
        "algorithm": "BiGDN",
        "description": "BiGDN node encoder weights",
        "filename": "node_encoder.pth",
        "url": f"{WEIGHTS_BASE_URL}/node_encoder.pth",
        "md5": None,
        "size_mb": 0.2,
    },
    "q_net_s.pth": {
        "algorithm": "BiGDNS",
        "description": "BiGDN student Q-network weights",
        "filename": "q_net_s.pth",
        "url": f"{WEIGHTS_BASE_URL}/q_net_s.pth",
        "md5": None,
        "size_mb": 0.1,
    },
    "tripling.ckpt": {
        "algorithm": "ToupleGDD",
        "description": "ToupleGDD Tripling model weights",
        "filename": "tripling.ckpt",
        "url": f"{WEIGHTS_BASE_URL}/tripling.ckpt",
        "md5": None,
        "size_mb": 1.2,
    },
    "s2vdqn.ckpt": {
        "algorithm": "S2V-DQN",
        "description": "S2V-DQN model weights",
        "filename": "s2vdqn.ckpt",
        "url": f"{WEIGHTS_BASE_URL}/s2vdqn.ckpt",
        "md5": None,
        "size_mb": 0.8,
    },
}


class WeightManager:
    """预训练权重管理器。

    管理深度强化学习算法的预训练权重，支持：
    - 从本地缓存目录加载
    - 从远程 URL 下载到缓存
    - 权重文件校验

    Example:
        >>> from pynetim.weights import WeightManager
        >>> 
        >>> # 获取权重路径
        >>> weights_path = WeightManager.get_weights_path("bigdn_weights.pth")
        >>> 
        >>> # 加载权重
        >>> import torch
        >>> state_dict = torch.load(weights_path)
    """

    _cache_dir: Path = WEIGHTS_CACHE_DIR
    _initialized: bool = False

    @classmethod
    def _ensure_cache_dir(cls) -> None:
        """确保缓存目录存在。"""
        if not cls._cache_dir.exists():
            cls._cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _compute_md5(cls, filepath: Path) -> str:
        """计算文件的 MD5 值。

        Args:
            filepath: 文件路径。

        Returns:
            str: MD5 哈希值。
        """
        md5_hash = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    @classmethod
    def _download_file(cls, url: str, dest: Path, description: str = "Downloading") -> None:
        """从 URL 下载文件。

        Args:
            url: 下载 URL。
            dest: 目标文件路径。
            description: 进度条描述。
        """
        cls._ensure_cache_dir()

        try:
            print(f"{description}: {url}")
            
            # 使用 tqdm 显示进度条
            response = urllib.request.urlopen(url)
            total_size = int(response.headers.get('content-length', 0))
            
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest.name) as pbar:
                with open(dest, 'wb') as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✓ Downloaded to: {dest}")
        except Exception as e:
            raise RuntimeError(f"Failed to download weights from {url}: {e}")

    @classmethod
    def get_weights_path(
        cls,
        weight_name: str,
        force_download: bool = False,
        verbose: bool = True,
    ) -> Path:
        """获取权重文件路径。

        查找顺序：
        1. 本地缓存目录 (~/.pynetim/weights/)
        2. 从远程 URL 下载到缓存

        Args:
            weight_name: 权重文件名或权重配置键。
            force_download: 强制从远程下载。
            verbose: 是否输出详细信息。

        Returns:
            Path: 权重文件的绝对路径。

        Raises:
            FileNotFoundError: 权重文件未找到。
            ValueError: 权重配置不存在。

        Example:
            >>> weights_path = WeightManager.get_weights_path("bigdn_weights.pth")
            >>> print(weights_path)
            /home/user/.pynetim/weights/bigdn_weights.pth
        """
        if weight_name not in WEIGHTS_CONFIG:
            raise ValueError(
                f"Unknown weight name: {weight_name}. "
                f"Available weights: {list(WEIGHTS_CONFIG.keys())}"
            )

        config = WEIGHTS_CONFIG[weight_name]
        filename = config["filename"]
        cache_file = cls._cache_dir / filename

        if cache_file.exists() and not force_download:
            if verbose:
                print(f"Using cached weights: {cache_file}")
            return cache_file

        if config.get("url"):
            cls._download_file(config["url"], cache_file, f"Downloading {weight_name}")
            return cache_file

        raise FileNotFoundError(
            f"Weights file '{weight_name}' not found.\n"
            f"Cache directory: {cls._cache_dir}\n"
            f"No download URL configured for this weight.\n"
            f"Please download the weights manually or configure a valid URL in WEIGHTS_CONFIG."
        )

    @classmethod
    def clear_cache(cls) -> None:
        """清空权重缓存目录。"""
        if cls._cache_dir.exists():
            shutil.rmtree(cls._cache_dir)
            print(f"Cleared weights cache: {cls._cache_dir}")

    @classmethod
    def list_cached_weights(cls) -> List[str]:
        """列出缓存目录中的所有权重文件。

        Returns:
            list: 权重文件名列表。
        """
        if not cls._cache_dir.exists():
            return []
        return [f.name for f in cls._cache_dir.iterdir() if f.is_file()]

    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """获取缓存信息。

        Returns:
            Dict: 包含缓存目录、文件数量、总大小等信息。
        """
        if not cls._cache_dir.exists():
            return {
                "cache_dir": str(cls._cache_dir),
                "exists": False,
                "num_files": 0,
                "total_size_mb": 0.0,
            }

        files = list(cls._cache_dir.iterdir())
        total_size = sum(f.stat().st_size for f in files if f.is_file())

        return {
            "cache_dir": str(cls._cache_dir),
            "exists": True,
            "num_files": len([f for f in files if f.is_file()]),
            "total_size_mb": total_size / (1024 * 1024),
        }

    @classmethod
    def set_cache_dir(cls, path: Path):
        """设置缓存目录。

        Args:
            path: 新的缓存目录路径。
        """
        cls._cache_dir = Path(path)
        cls._ensure_cache_dir()
