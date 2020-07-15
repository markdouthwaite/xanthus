from setuptools import setup
from setuptools import find_packages

from xanthus import __version__

long_description = """
Implementations of Neural Collaborative Filtering models, based on/inspired by the work 
of He et al., built on Keras, with a few recommendation system utilities to boot. 
"""

setup(
    name="Xanthus",
    version=__version__,
    description="Neural Collaborative Filtering in Python",
    long_description=long_description,
    author="Mark Douthwaite",
    author_email="mark@douthwaite.io",
    url="https://github.com/markdouthwaite/xanthus",
    license="MIT",
    install_requires=["tensorflow", "pandas", "numpy", "h5py"],
    extras_require={"tests": ["pytest", "pandas", "requests", "markdown", "black"],},
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
