import pathlib


NOTEBOOK_GLOB = "*.ipynb"
LEGACY_PATTERNS = [
    "from tqdm.notebook import tqdm",
    "m_values, positions, internal_amplitude, internal_is_ground = make_atom_states(",
    "m_values, squiggly_amplitudes, internal_is_ground, positions = pulse_interaction_in_borde_representation(",
    "vs.tag_plot(ax=ax, small=True)",
]


def test_notebooks_do_not_use_removed_api_shapes():
    root = pathlib.Path(__file__).resolve().parent
    notebooks = sorted(root.glob(NOTEBOOK_GLOB))
    assert notebooks, "No notebooks found to validate"

    for notebook in notebooks:
        text = notebook.read_text(encoding="utf-8")
        for pattern in LEGACY_PATTERNS:
            assert pattern not in text, f"{notebook.name} still contains legacy pattern: {pattern}"
