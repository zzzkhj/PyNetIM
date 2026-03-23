import os
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

cpp_dir = os.path.join("src", "pynetim", "cpp")

ext_modules = [
    Pybind11Extension(
        "pynetim.cpp.graph.graph",
        [os.path.join(cpp_dir, "bindings", "graph_bind.cpp")],
        include_dirs=[os.path.join(cpp_dir, "include")],
        cxx_std=20,
        extra_compile_args=["-mavx2"],
    ),
    Pybind11Extension(
        "pynetim.cpp.diffusion_model.independent_cascade_model",
        [
            os.path.join(cpp_dir, "bindings", "ic_bind.cpp"),
            os.path.join(cpp_dir, "bindings", "graph_bind.cpp"),
        ],
        include_dirs=[os.path.join(cpp_dir, "include")],
        cxx_std=20,
        extra_compile_args=["-mavx2"],
    ),
    Pybind11Extension(
        "pynetim.cpp.diffusion_model.linear_threshold_model",
        [
            os.path.join(cpp_dir, "bindings", "lt_bind.cpp"),
            os.path.join(cpp_dir, "bindings", "graph_bind.cpp"),
        ],
        include_dirs=[os.path.join(cpp_dir, "include")],
        cxx_std=20,
        extra_compile_args=["-mavx2"],
    ),
]

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)