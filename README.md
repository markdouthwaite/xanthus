<h1 align="center">Xanthus: Neural Recommendation Models in Python</h1>

<p align="center">
<a href="https://github.com/markdouthwaite/xanthus/actions"><img alt="Build: Unknown" src="https://github.com/markdouthwaite/xanthus/workflows/Build/badge.svg"></a>
<img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-000000.svg">
</p>

[**Quickstart**](#quickstart)
| [**Install guide**](#installation)
| [**Known issues**](#known-issues)

## What is Xanthus?

Xanthus is a package that provides the tools and model architectures necessary to 
utilise the techniques outlined in [He et al's work]() on Neural Collaborative Filtering in 
your own projects. Over time, you'll find work from other research finding it's way in
here. The aim of this package is to make state-of-the-art research into neural 
recommenders accessible and ultimately ready for deployment. You might find that it's 
not quite there yet on that latter point, but it'll get there eventually.

Sound good? Great, here goes.

## Quickstart

So you want to get straight into the code? Here's a [Colab notebook]() just for you.

You can also find some examples of how to use Xanthus with sample datasets in this
repo's docs. These include:

* [A minimal example using the Movielens (100k) dataset.]()
* [An example using the meta-data features of Xanthus on the Movielens (100k) dataset.]()

If you're interested in seeing the contents of this package benchmarked against the 
non-neural recommendation packages, you're in luck. Here's some [benchmarking results]() 
for you to peruse at your leisure.

## Installation

Xanthus is a pure Python package, and you'll need Python 3.6 or greater to use Xanthus.

To install, simply run:

```bash
pip install xanthus
```

That's it, you're good to go. Well, except for one thing...

The package makes extensive use of [Tensorflow 2.0]() and the [Keras]() API. If
you want to make use of the GPU acceleration provided by Tensorflow, you'll need to 
follow the [Tensorflow team's guide]() for setting that up. If you don't need GPUs
right now, then great, you _really_ are all set.

## Walkthrough

### Data preparation

### Model setup

### Model training

## Experimental results

## Known issues
