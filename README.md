<h1 align="center">Xanthus</h1>

<p align="center">
<a href="https://github.com/markdouthwaite/xanthus/actions"><img alt="Build: Unknown" src="https://github.com/markdouthwaite/xanthus/workflows/Build/badge.svg"></a>
<img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-000000.svg">
</p>

You'll need Python 3.6 or greater to use Xanthus.

## Todo

1. Implement Keras-based 'model wrapper'.
    a. For GMF
    b. For MLP
    c. For NMF
2. Implement Implicit-based 'model wrapper' for benchmarking.
3. Implement a PopRank model with the same API as 'model wrapper'.
4. Run benchmark tests to corroborate paper/s.
    a. For GMF
    b. For MLP
    c. For NMF
    d. For PopRank
    e. For BPR (Implicit)
    f. For ALS (Implicit)
5. Include the benchmarks as a script & notebook. Run on MovieLens and one other dataset.
    a. Add colab notebook.
7. Create `xanthus-service` template with:
    a. A model server built on FastAPI.
    b. An app for visualizing recommendations.
    c. An example deployment for Kubernetes.
8. Write docs:
    0. Recommendations
    a. Getting started
    b. Deploying to GCP
