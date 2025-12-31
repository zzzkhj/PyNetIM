from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "pynetim.cpp.graph.graph",
        ["src/pynetim/cpp/graph/graph_bind.cpp"],
        include_dirs=["src/pynetim/cpp/graph"],
        cxx_std=20,
    ),
    Pybind11Extension(
        "pynetim.cpp.diffusion_model.independent_cascade_model",
        [
            "src/pynetim/cpp/diffusion_model/ic_bind.cpp",
            "src/pynetim/cpp/graph/graph_bind.cpp",
        ],
        include_dirs=[
            "src/pynetim/cpp/diffusion_model",
            "src/pynetim/cpp/graph",
        ],
        cxx_std=20,
    ),
    Pybind11Extension(
        "pynetim.cpp.diffusion_model.linear_threshold_model",
        [
            "src/pynetim/cpp/diffusion_model/lt_bind.cpp",
            "src/pynetim/cpp/graph/graph_bind.cpp",
        ],
        include_dirs=[
            "src/pynetim/cpp/diffusion_model",
            "src/pynetim/cpp/graph",
        ],
        cxx_std=20,
    ),
]

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)