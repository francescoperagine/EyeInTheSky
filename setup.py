from setuptools import find_packages, setup

setup(
    name='eye_in_the_sky',
    packages=find_packages("eyeinthesky"),
    package_dir={"": "eyeinthesky"},
    version='0.1.0',
    description='Deep Learning-Driven Automated Detection of Traffic Accidents in Aerial Imagery',
    author='Francesco Peragine',
    license='MIT'
)