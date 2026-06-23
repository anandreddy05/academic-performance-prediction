from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as f:
        return [line.strip() for line in f if line.strip() != "-e ."]


print(get_requirements("./requirements.txt"))
setup(
    name="academic-performance-prediction",
    version="1",
    author="Anand",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
