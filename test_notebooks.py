import nbformat
import nbclient
import os
import pytest

NOTEBOOK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")


def get_all_notebooks():
    return sorted(
        os.path.join(NOTEBOOK_DIR, f)
        for f in os.listdir(NOTEBOOK_DIR)
        if f.endswith(".ipynb")
    )


@pytest.mark.parametrize("notebook_path", get_all_notebooks())
def test_notebook_execution(notebook_path):
    """Test that a Jupyter notebook runs without errors."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = nbclient.NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": NOTEBOOK_DIR}},
    )
    try:
        client.execute()
    except nbclient.exceptions.CellExecutionError as e:
        pytest.fail(f"Notebook {notebook_path} failed to execute: {e}")
