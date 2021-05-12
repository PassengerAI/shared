from setuptools import setup, find_packages

setup(
    name='pai-datatypes',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6, <3.7',
    install_requires=[
        'dataclasses',
        'numpy',
        'scipy'
    ]
)
