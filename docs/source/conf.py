# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyNetIM'
copyright = '2025, Kaijing Zhang'
author = 'Kaijing Zhang'
release = 'v0.2.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
import sys

# 添加项目路径到 sys.path
sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.autodoc',           # 自动从 docstring 生成文档
    'sphinx.ext.napoleon',          # 支持 Google/NumPy 风格 docstring
    'sphinx.ext.viewcode',          # 添加源码链接
    'sphinx.ext.githubpages',       # GitHub Pages 支持
    'autoapi.extension',            # 自动 API 文档生成
]

# AutoAPI 配置（自动扫描所有模块）
autoapi_dirs = ['../../src/pynetim']
autoapi_type = 'python'
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'imported-members',
]
autoapi_ignore = ['*/__pycache__/*', '*/tests/*']

# Napoleon 设置
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

templates_path = ['_templates']
exclude_patterns = []

# 语言设置
language = 'zh_CN'  # 或 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# 主题选项
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
}
