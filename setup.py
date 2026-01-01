"""
Setup configuration for AMR Path Planner package.
"""

from pathlib import Path
from setuptools import setup, find_packages

BASE_DIR = Path(__file__).resolve().parent


def read_readme() -> str:
    """Read README.md safely."""
    readme_path = BASE_DIR / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "Autonomous Mobile Robot Path Planner with dynamic obstacle avoidance."


def read_requirements() -> list[str]:
    """Read requirements.txt safely."""
    req_path = BASE_DIR / "requirements.txt"
    if not req_path.exists():
        return ["numpy", "matplotlib", "networkx"]

    requirements: list[str] = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements


setup(
    name="amr-path-planner",
    version="1.0.0",
    author="Akshay Patel",
    author_email="ap00143@mix.wvu.edu",
    description="Autonomous Mobile Robot Path Planner with dynamic obstacle avoidance",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/aksh-ay06/Autonomous-Mobile-Robot-Path-Planner",
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*", "examples", "examples.*")),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "isort>=5.0",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "Topic :: Scientific/Engineering :: Robotics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "robotics",
        "path-planning",
        "autonomous-robots",
        "a-star",
        "rrt",
        "rrt-star",
        "prm",
        "motion-planning",
        "navigation",
    ],
    project_urls={
        "Source": "https://github.com/aksh-ay06/Autonomous-Mobile-Robot-Path-Planner",
        "Issues": "https://github.com/aksh-ay06/Autonomous-Mobile-Robot-Path-Planner/issues",
        "Documentation": "https://github.com/aksh-ay06/Autonomous-Mobile-Robot-Path-Planner#readme",
    },
)
