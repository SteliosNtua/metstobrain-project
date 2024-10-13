from setuptools import setup, find_packages

setup(
    name="v2",
    package_dir={'': 'v2'},
    packages=find_packages(
        'v2', include=('v2/modules',), exclude=('tests',)),
)