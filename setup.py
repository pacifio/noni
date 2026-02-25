from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")

VERSION = "0.1.0"

setup(
    name="noniml",
    version=VERSION,
    description="Noni — a tiny tensor library with autograd, for humans.",
    long_description=long_description,
    long_description_content_type="text/markdown",

    author="Adib Mohsin",
    author_email="adibmohsin.root@gmail.com",
    url="https://github.com/pacifio/noni",

    license="MIT",

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],

    packages=find_packages(exclude=["tests*", "projects*", "docs*", "venv*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
    ],

    extras_require={
        "opencl": ["pyopencl"],
        "all":    ["pyopencl"],
        "dev":    ["pytest", "build", "twine"],
    },
    keywords=["autograd", "tensor", "deep-learning", "machine-learning", "numpy", "educational"],
    project_urls={
        "Source": "https://github.com/pacifio/noni",
        "Bug Reports": "https://github.com/pacifio/noni/issues",
    },
)
