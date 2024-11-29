# -*- coding: utf-8 -*-
"""Setup script."""
import pathlib
import setuptools

from typing import Union

import sys
import importlib.util


def import_from_path(path):
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Optional: add to sys.modules
    spec.loader.exec_module(module)
    return module


def get_req(file_path: Union[str, pathlib.Path]) -> list[str]:
    """Retrieve requirements from a pip-requirements file"""
    with open(file_path, "r") as f:
        reqs = [str(req) for req in f.readlines()]
    return reqs


project_path = pathlib.Path(__file__).parent

version_module = import_from_path(project_path / "src" / "common" / "__version__.py")
metadata = version_module.__dict__

with open(project_path / "README.md") as file:
    reamde = file.read()

if __name__ == "__main__":
    setuptools.setup(
        name=metadata["__title__"],
        description=metadata["__description__"],
        version=metadata["__version__"],
        author=metadata["__author__"],
        url=metadata["__url__"],
        license=metadata["__license__"],
        license_files=["LICENSE"],
        long_description=reamde,
        long_description_content_type="text/markdown",
        packages=["common"] + setuptools.find_packages(where="src"),
        package_dir={"": "src"},
        package_data={
            "common": ["objects/**/*", "data/**/*"]
        },
        platforms=["unix", "linux", "cygwin", "win32"],
        python_requires=">=3.11",
        install_requires=get_req(file_path=project_path / "requirements.txt"),
        # additional requirements: to be isntall like '.[dev]'
        extras_require={
            # requirements for linting
            "dev": get_req(file_path=project_path / "dev-requirements.txt")
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Researchers",
            "Natural Language :: English",
            "License :: OSI Approved :: TBD",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.12",
        ],
    )
