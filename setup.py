from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "A modular Retrieval-Augmented Generation (RAG) repository"

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="rag-master-repo",
    version="0.1.0",
    author="Renswick Delvar",
    author_email="renswick.delvar@gmail.com",
    description="A modular Retrieval-Augmented Generation (RAG) repository with pluggable components",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/renswickd/rag-master-repo",
    keywords="rag, retrieval-augmented-generation, langchain, chromadb, groq, langgraph, agentic-rag, rag-ubac, cache-rag",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    include_package_data=True,
    license="MIT",
    license_files=["LICENSE"],
)
