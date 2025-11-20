#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
12D Cosmic Synapse Transformer - Setup Configuration

This setup.py makes the package pip-installable with:
    pip install -e .

Author: Cory Shane Davis
License: MIT
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Define package metadata
PACKAGE_NAME = "cosmic-synapse-transformer"
VERSION = "1.0.0"
DESCRIPTION = "12D Cosmic Synapse Transformer - Revolutionary AI Architecture"
AUTHOR = "Cory Shane Davis"
AUTHOR_EMAIL = "cory@cosmicsynapse.ai"
URL = "https://github.com/NavisWORLD/infinite-adaptive-audio-12d-universe-engine"
LICENSE = "MIT"

# Core dependencies
INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "transformers>=4.30.0",
    "tqdm>=4.65.0",
    "matplotlib>=3.7.0",
    "scipy>=1.10.0",
    "pyyaml>=6.0.0",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=7.3.1',
        'pytest-cov>=4.1.0',
        'black>=23.3.0',
        'flake8>=6.0.0',
        'mypy>=1.3.0',
        'jupyter>=1.0.0',
        'ipython>=8.14.0',
    ],
    'api': [
        'flask>=2.3.2',
        'flask-cors>=4.0.0',
        'gunicorn>=20.1.0',
    ],
    'monitoring': [
        'wandb>=0.15.4',
        'tensorboard>=2.13.0',
    ],
    'all': [
        'pytest>=7.3.1',
        'pytest-cov>=4.1.0',
        'black>=23.3.0',
        'flake8>=6.0.0',
        'mypy>=1.3.0',
        'jupyter>=1.0.0',
        'ipython>=8.14.0',
        'flask>=2.3.2',
        'flask-cors>=4.0.0',
        'gunicorn>=20.1.0',
        'wandb>=0.15.4',
        'tensorboard>=2.13.0',
    ]
}

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
]

# Keywords
KEYWORDS = [
    "transformer",
    "deep-learning",
    "neural-networks",
    "ai",
    "machine-learning",
    "cosmic-synapse",
    "12d-theory",
    "golden-ratio",
    "consciousness",
    "hebbian-learning",
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    packages=find_packages(exclude=['tests', 'docs', 'examples', 'notebooks']),
    py_modules=[
        'cosmic_synapse_transformer',
        'train_cosmic_transformer',
        'inference_cosmic_transformer',
        'demo_cosmic_transformer',
    ],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires='>=3.8',
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    entry_points={
        'console_scripts': [
            'cosmic-train=train_cosmic_transformer:main',
            'cosmic-infer=inference_cosmic_transformer:main',
            'cosmic-demo=demo_cosmic_transformer:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        'Bug Reports': f'{URL}/issues',
        'Source': URL,
        'Documentation': f'{URL}#readme',
    },
)
