import setuptools

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "1.0.0"

# --- Project Metadata ---
REPO_NAME = "llm-model-analyzer"
AUTHOR_USER_NAME = "PUNAMRAMPUKALE"
SRC_REPO = "mlProject"
AUTHOR_EMAIL = "punampukale@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="LLM Response Quality Analyzer — a FastAPI-based ML service for evaluating and visualizing LLM response quality using programmatic metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
        "Documentation": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}#readme",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn>=0.32.0",
        "orjson>=3.10.0",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "textstat>=0.7.0",
        "nltk>=3.9.0",
        "spacy>=3.7.0",
        "sentence-transformers>=2.2.0",
        "torch>=2.2.0",
        "pydantic>=2.8.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0"
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Framework :: FastAPI",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm metrics analyzer fastapi embeddings coherence readability ml quality-eval",
)
