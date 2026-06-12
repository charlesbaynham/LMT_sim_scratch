import nbformat
import nbclient
import os
import pytest

NOTEBOOK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")

# Marker emitted by the guard cell of a deliberately parked notebook (one fenced
# off behind ``raise NotImplementedError`` because it runs on legacy/placeholder
# data). Such notebooks are EXPECTED to raise rather than execute cleanly.
PARKED_MARKER = "PARKED: do not run this notebook"


def get_all_notebooks():
    return sorted(
        os.path.join(NOTEBOOK_DIR, f)
        for f in os.listdir(NOTEBOOK_DIR)
        if f.endswith(".ipynb")
    )


def _is_parked(nb):
    return any(
        cell.cell_type == "code" and PARKED_MARKER in cell.source
        for cell in nb.cells
    )


@pytest.mark.parametrize("notebook_path", get_all_notebooks())
def test_notebook_execution(notebook_path):
    """Test that a Jupyter notebook runs without errors.

    Notebooks deliberately parked behind a ``raise NotImplementedError`` guard
    are the exception: they must raise that guard (and nothing else) instead of
    executing cleanly.
    """
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = nbclient.NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": NOTEBOOK_DIR}},
    )

    if _is_parked(nb):
        with pytest.raises(nbclient.exceptions.CellExecutionError) as excinfo:
            client.execute()
        assert "NotImplementedError" in str(excinfo.value), (
            f"Parked notebook {notebook_path} raised an unexpected error "
            f"instead of its NotImplementedError guard: {excinfo.value}"
        )
        return

    try:
        client.execute()
    except nbclient.exceptions.CellExecutionError as e:
        pytest.fail(f"Notebook {notebook_path} failed to execute: {e}")

