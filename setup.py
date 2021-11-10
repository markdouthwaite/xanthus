from setuptools import setup
from setuptools import find_packages

from xanthus import __version__

long_description = """
Implementations of Neural Collaborative Filtering models, based on/inspired by the work 
of He et al., built on Keras, with a few recommendation system utilities to boot. 
"""

setup(
    name="xanthus",
    version=__version__,
    description="Neural Collaborative Filtering in Python",
    long_description=long_description,
    author="Mark Douthwaite",
    author_email="mark@douthwaite.io",
    url="https://github.com/markdouthwaite/xanthus",
    license="MIT",
    install_requires=["tensorflow==2.5.2", "jupyter==1.0.0", "pandas==1.0.4", "numpy==1.18.5", "h5py==2.10.0", "scipy==1.4.1", "scikit-learn==0.23.1", "requests==2.23.0", "implicit==0.4.0"],
    extras_require={"tests": ["pytest", "pandas", "requests", "markdown", "black", "tensorflow"]},
    classifiers=[
        # 'Development Status :: 5 - Production/Stable',
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
)
