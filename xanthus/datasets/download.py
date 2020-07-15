import os
import io
import requests
import zipfile


def movielens(
    version: str = "latest-small",
    base_url: str = "http://files.grouplens.org/datasets/movielens/ml-{version}.zip",
    output_dir: str = "data",
    unzip: bool = True,
) -> None:
    """Download a given movielens dataset."""

    response = requests.get(base_url.format(version=version))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if unzip:
        zipfile.ZipFile(io.BytesIO(response.content)).extractall(output_dir)
    else:
        with open(os.path.join(output_dir, f"ml-{version}", "wb")) as file:
            file.write(response.content)
