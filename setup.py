import os
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

cpp_dir = os.path.join("src", "pynetim", "cpp")
include_dir = os.path.join(cpp_dir, "include")
bindings_dir = os.path.join(cpp_dir, "bindings")

ext_modules = [
    Pybind11Extension(
        "pynetim.utils.utils",
        [os.path.join(bindings_dir, "utils_bind.cpp")],
        include_dirs=[bindings_dir, include_dir],
        cxx_std=20,
        extra_compile_args=["-mavx2"],
    ),
    Pybind11Extension(
        "pynetim.graph.graph",
        [os.path.join(bindings_dir, "graph", "graph_bind.cpp")],
        include_dirs=[bindings_dir, include_dir],
        cxx_std=20,
        extra_compile_args=["-mavx2"],
    ),
    Pybind11Extension(
        "pynetim.diffusion_model.independent_cascade_model",
        [
            os.path.join(bindings_dir, "diffusion_model", "ic_bind.cpp"),
            os.path.join(bindings_dir, "graph", "graph_bind.cpp"),
        ],
        include_dirs=[bindings_dir, include_dir],
        cxx_std=20,
        extra_compile_args=["-mavx2"],
    ),
    Pybind11Extension(
        "pynetim.diffusion_model.linear_threshold_model",
        [
            os.path.join(bindings_dir, "diffusion_model", "lt_bind.cpp"),
            os.path.join(bindings_dir, "graph", "graph_bind.cpp"),
        ],
        include_dirs=[bindings_dir, include_dir],
        cxx_std=20,
        extra_compile_args=["-mavx2"],
    ),
    Pybind11Extension(
        "pynetim.diffusion_model.susceptible_infected_model",
        [
            os.path.join(bindings_dir, "diffusion_model", "si_bind.cpp"),
            os.path.join(bindings_dir, "graph", "graph_bind.cpp"),
        ],
        include_dirs=[bindings_dir, include_dir],
        cxx_std=20,
        extra_compile_args=["-mavx2"],
    ),
    Pybind11Extension(
        "pynetim.diffusion_model.susceptible_infected_recovered_model",
        [
            os.path.join(bindings_dir, "diffusion_model", "sir_bind.cpp"),
            os.path.join(bindings_dir, "graph", "graph_bind.cpp"),
        ],
        include_dirs=[bindings_dir, include_dir],
        cxx_std=20,
        extra_compile_args=["-mavx2"],
    ),
    Pybind11Extension(
        "pynetim.diffusion_model.py_diffusion_model_base",
        [
            os.path.join(bindings_dir, "diffusion_model", "py_diffusion_model_bind.cpp"),
            os.path.join(bindings_dir, "graph", "graph_bind.cpp"),
        ],
        include_dirs=[bindings_dir, include_dir],
        cxx_std=20,
        extra_compile_args=["-mavx2"],
    ),
    Pybind11Extension(
        "pynetim.algorithms.ris.base_ris_algorithm",
        [os.path.join(bindings_dir, "algorithms", "base_ris_bind.cpp")],
        include_dirs=[bindings_dir, include_dir],
        cxx_std=20,
        extra_compile_args=["-mavx2"],
    ),
    Pybind11Extension(
        "pynetim.algorithms.ris.imm_algorithm",
        [os.path.join(bindings_dir, "algorithms", "imm_bind.cpp")],
        include_dirs=[bindings_dir, include_dir],
        cxx_std=20,
        extra_compile_args=["-mavx2"],
    ),
    Pybind11Extension(
        "pynetim.algorithms.ris.tim_algorithm",
        [os.path.join(bindings_dir, "algorithms", "tim_bind.cpp")],
        include_dirs=[bindings_dir, include_dir],
        cxx_std=20,
        extra_compile_args=["-mavx2"],
    ),
    Pybind11Extension(
        "pynetim.algorithms.ris.opim_algorithm",
        [os.path.join(bindings_dir, "algorithms", "opim_bind.cpp")],
        include_dirs=[bindings_dir, include_dir],
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
