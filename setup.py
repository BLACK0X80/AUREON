from setuptools import setup, find_packages
import os


def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]


setup(
    name="aureon",
    version="1.0.0",
    author="AUREON Team",
    author_email="contact@aureon.ai",
    description="A comprehensive AI/ML pipeline management system",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/aureon/aureon",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aureon=aureon.cli.main:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "aureon": [
            "config/*.yaml",
            "config/*.json",
        ],
    },
    keywords=[
        "machine learning",
        "artificial intelligence",
        "data science",
        "mlops",
        "pipeline",
        "automation",
        "model management",
        "api",
        "fastapi",
        "scikit-learn",
    ],
    project_urls={
        "Bug Reports": "https://github.com/aureon/aureon/issues",
        "Source": "https://github.com/aureon/aureon",
        "Documentation": "https://aureon.readthedocs.io/",
        "Changelog": "https://github.com/aureon/aureon/blob/main/CHANGELOG.md",
    },
    zip_safe=False,
)
