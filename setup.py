"""Setup configuration for pneumonia detection package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (this_directory / "requirements.txt").read_text(encoding="utf-8").split("\n")
requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="pneumonia-detection",
    version="1.0.0",
    author="Anurag",
    author_email="anuragverma08002@gmail.com",
    description="Professional deep learning system for automated pneumonia detection from chest X-ray images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pneumonia-detection",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/pneumonia-detection/issues",
        "Documentation": "https://github.com/yourusername/pneumonia-detection#readme",
        "Source Code": "https://github.com/yourusername/pneumonia-detection",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pneumonia-detection=main:main",
        ],
    },
    include_package_data=True,
    keywords=[
        "pneumonia",
        "detection",
        "deep learning",
        "medical imaging",
        "x-ray",
        "cnn",
        "tensorflow",
        "keras",
    ],
)
