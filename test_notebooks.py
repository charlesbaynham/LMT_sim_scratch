import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import pytest

# Define the directory containing the notebooks
NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))


# Collect all notebooks in the directory
def get_all_notebooks():
    notebooks = []
    for file in os.listdir(NOTEBOOK_DIR):
        if file.endswith(".ipynb"):
            notebooks.append(os.path.join(NOTEBOOK_DIR, file))
    return notebooks


# Parametrize the test to run for each notebook
@pytest.mark.parametrize("notebook_path", get_all_notebooks())
def test_notebook_execution(notebook_path):
    """Test that a Jupyter notebook runs without errors."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    try:
        ep.preprocess(nb, {"metadata": {"path": NOTEBOOK_DIR}})
    except Exception as e:
        pytest.fail(f"Notebook {notebook_path} failed to execute: {e}")
