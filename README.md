<h1 align="center">Xanthus: Neural Recommendation Models in Python</h1>

<p align="center">
<a href="https://github.com/markdouthwaite/xanthus/actions"><img alt="Build: Unknown" src="https://github.com/markdouthwaite/xanthus/workflows/Build/badge.svg"></a>
<img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-000000.svg">
</p>

[**Quickstart**](#quickstart)
| [**Install guide**](#installation)

## What is Xanthus?

Xanthus is a package that was started as personal project to translate an academic ML paper into a 'production ready' software package. It provides the tools and model architectures necessary to utilise the techniques outlined in [He et al's work](https://dl.acm.org/doi/10.1145/3038912.3052569) on Neural Collaborative Filtering in your own projects. Over time, you'll find work from other research finding it's way in here. The aim of this package is to make state-of-the-art research into neural recommenders accessible and ultimately ready for deployment. You might find that it's not quite there yet on that latter point, but sit tight, it will be.

Sound good? Great, here goes.

## Quickstart

Want to get straight into the code? Here's an [introductory notebook](docs/getting-started.ipynb) just for you.

You can also find some examples of how to use Xanthus with sample datasets in this
repo's docs. These include:

* [A minimal example using the Movielens (100k) dataset.](docs/getting-started.ipynb)
* [An example using the meta-data features of Xanthus on the Movielens (100k) dataset.](examples/metadata.py)

If you're interested in seeing the results of benchmarking of Xanthus' models against 'classic' 
collaborative filtering models, you'll have to sit tight. But rest assured: benchmarks are on their way.

## Installation

Xanthus is a pure Python package, and you'll need Python 3.6 or greater to use Xanthus.

To install, simply run:

```bash
pip install xanthus
```

That's it, you're good to go. Well, except for one thing...

The package makes extensive use of [Tensorflow 2.0](https://www.tensorflow.org/tutorials/quickstart/beginner) and the [Keras](https://keras.io/) API. If
you want to make use of the GPU acceleration provided by Tensorflow, you'll need to 
follow the [Tensorflow team's guide](https://www.tensorflow.org/install/gpu) for setting that up. If you don't need GPUs
right now, then great, you _really_ are all set.
