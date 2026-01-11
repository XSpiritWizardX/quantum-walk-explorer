from pathlib import Path

from setuptools import find_packages, setup


README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")


setup(
    name="quantum-walk-explorer",
    version="0.1.0",
    description="Quantum walk visualizer and logging toolkit",
    long_description=README,
    long_description_content_type="text/markdown",
    requires_python=">=3.11",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24",
        "matplotlib>=3.7",
        "imageio>=2.31",
        "pillow>=8.3.2",
        "flask>=2.3",
        "gunicorn>=21.2",
    ],
    entry_points={
        "console_scripts": [
            "qwe=quantum_walk_explorer.cli:main",
        ]
    },
)
