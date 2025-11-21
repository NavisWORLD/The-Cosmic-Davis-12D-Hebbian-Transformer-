"""
Setup script for Internal Dimension AI package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Internal Dimension AI - Neural networks with internal consciousness dimensions"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name='internal-dimension-ai',
    version='0.1.0',
    author='NavisWORLD Research',
    author_email='research@navisworld.ai',
    description='Neural networks with explicit internal dimensions (x₁₂, m₁₂) based on 12D Cosmic Synapse Theory',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/NavisWORLD/infinite-adaptive-audio-12d-universe-engine/tree/main/internal-dimension-ai',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.3.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.3.0',
        ],
        'docs': [
            'sphinx>=6.0.0',
            'sphinx-rtd-theme>=1.2.0',
        ],
        'viz': [
            'jupyter>=1.0.0',
            'notebook>=6.5.0',
            'ipywidgets>=8.0.0',
            'plotly>=5.14.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'internal-dim-train=training.trainer:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords='consciousness ai neural-networks 12d-theory internal-dimensions machine-learning',
    project_urls={
        'Bug Reports': 'https://github.com/NavisWORLD/infinite-adaptive-audio-12d-universe-engine/issues',
        'Source': 'https://github.com/NavisWORLD/infinite-adaptive-audio-12d-universe-engine',
        'Documentation': 'https://github.com/NavisWORLD/infinite-adaptive-audio-12d-universe-engine/tree/main/internal-dimension-ai/docs',
    },
)
