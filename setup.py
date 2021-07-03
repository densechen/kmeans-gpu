'''
python setup.py sdist bdist_wheel
python -m twine upload dist/*
'''

from setuptools import find_packages, setup

from kmeans_gpu import __version__

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    install_requires=install_requires,
    name="kmeans_gpu",
    version=__version__,
    author="densechen",
    author_email="densechen@foxmail.com",
    description="KMeans-GPU: A PyTorch Module for KMeans.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/densechen/kmeans-gpu",
    download_url='https://github.com/densechen/kmeans-gpu/archive/main.zip',
    packages=find_packages(),
    # https://pypi.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Documentation :: Sphinx",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="MIT License",
    python_requires='>=3.7',
)