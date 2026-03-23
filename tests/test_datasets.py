#pytest for Datasets

# Test PrecomputeDataset
# First smiles in first candidate list
# is:
# CC(C)[C@@H]1C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O1)CC2=CC=CC=C2)C)C(C)C)CC3=CC=CC=C3)C)C(C)C)CC4=CC=CC=C4)C
# Test Spectum Dataset 
# first spectrum's identifier is:
# MassSpecGymID0000201
# test_precompute_dataset.py
from .context import datasets as ds
from pathlib import Path

def test_precompute_candidate_dataset(tmp_path):
    # Path to your generated candidate JSON
    cand_pth = Path("data/sample/list_of_lists_of_candidates.json")

    # Expected first candidate
    expected_first = (
        "CC(C)[C@@H]1C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H]"
        "(C(=O)N([C@H](C(=O)O1)CC2=CC=CC=C2)C)C(C)C)CC3=CC=CC=C3)C)C(C)C)"
        "CC4=CC=CC=C4)C"
    )

    # Instantiate dataset
    dataset = ds.PrecomputeCandDataset(raw_pth=cand_pth)

    # Check first candidate
    assert dataset.cand_smiles[0] == expected_first

    # Check dataset length > 0
    assert len(dataset) > 0

# test_spec_dataset.py

def test_spec_dataset(tmp_path):
    # Path to your generated spectra JSON
    spec_pth = Path("data/sample/expanded_retrieval_dataset.json")

    # Expected first identifier
    expected_id = "MassSpecGymID0000201"

    # Instantiate dataset
    dataset = ds.SpecDataset(raw_pth=spec_pth)
    # First spectrum item
    first = dataset[0]

    # Check identifier
    assert first["spec"].metadata["identifier"] == expected_id

    # Check dataset length > 0
    assert len(dataset) > 0