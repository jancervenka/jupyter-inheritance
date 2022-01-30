import setuptools
import pathlib

VERSION = "0.1.0"

INSTALL_REQUIRES = [

    "dill>=0.3.4",
    "jupyter_client>=7.0.6",
    "jupyter_core>=4.9.1",
    "jupyter_server>=1.11.2",
    "requests",
]

EXTRAS_REQUIRE = {"tests": ["pytest"]}


def run_setup():
    """
    Runs the package setup.
    """

    this_directory = pathlib.Path(__file__).parent

    setup_params = {
        "name": "jupyter-inheritance",
        "version": VERSION,
        "description": "Inherit from Jupyter kernels",
        "author": "Jan Cervenka",
        "author_email": "jan.cervenka@yahoo.com",
        "long_description": (this_directory / "README.md").read_text(),
        "long_description_content_type": "text/markdown",
        "package_dir": {"": "src"},
        "packages": setuptools.find_packages(where="src"),
        "python_requires": ">=3.7",
        "install_requires": INSTALL_REQUIRES,
        "extras_require": EXTRAS_REQUIRE,
    }
    setuptools.setup(**setup_params)


if __name__ == "__main__":
    run_setup()
