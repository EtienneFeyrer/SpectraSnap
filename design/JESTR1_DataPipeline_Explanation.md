# JESTR1 Data Pipeline Documentation

## Overview

This document explains how the MassSpecGym dataset, retrieval candidates JSON file, and DataModules interact within the JESTR1 project to produce batches and final results.

---

## 1. Dataset Loading

### 1.1 Base Dataset: `MassSpecDataset`
**File:** `massspecgym/data/datasets.py`

The base dataset class loads mass spectrometry data from a TSV file:

```python
# Example TSV structure (data.tsv):
# identifier         | mzs                  | intensities        | fold  | precursor_mz
# MassSpecGymID0132861 | "95.04,107.03,..."  | "0.043,0.039,..." | train | 226.07
```

**Loading Process:**
1. `pd.read_csv(pth, sep="\t")` → metadata DataFrame
2. For each row, create a `matchms.Spectrum` object:
   ```python
   matchms.Spectrum(
       mz=np.array([95.04, 107.03, ...]),
       intensities=np.array([0.043, 0.039, ...]),
       metadata={"precursor_mz": 226.07}
   )
   ```
3. Drop `mzs` and `intensities` columns from metadata (stored in `spectra`)
4. Compute `mol_freq` (molecular frequency) by grouping on InChIKey

### 1.2 Extended Dataset: `JESTR1_MassSpecDataset`
**File:** `jestr/data/datasets.py`

Extends `MassSpecDataset` with:
- `spectra_view` parameter to select transform type

**`__getitem__(i)` returns:**
```python
{
    'SpecBinnerLog': Tensor[1005],     # binned spectrum
    'mol': DGLGraph,                    # molecular graph
    'mol_freq': int,                    # how often this molecule appears
    'identifier': 'MassSpecGymID0132861'
}
```

---

## 2. JSON Retrieval Candidates

### 2.1 File Structure
**Example:** `identifier_to_candidates.json`

```json
{
    "MassSpecGymID0132861": [
        "COC1=CC2=C(C=C1)N(C(=O)C(O2)O)OC",
        "CC(C)(C)SC(=S)Nc1ccccc1",
        "COC(=O)Cc1c(OC)cccc1[N+](=O)[O-]",
        // ... typically 100-500 candidate SMILES
    ],
    "MassSpecGymID0193514": [
        // ... candidates for this spectrum
    ]
}
```

### 2.2 How JSON is Used: `ExpandedRetrievalDataset`
**File:** `jestr/data/datasets.py`

**Initialization Process:**
1. Load candidates JSON: `json.load(file)` → `self.candidates`
2. Filter out candidates with '.' (mixtures)
3. Expand to spectrum-candidate pairs:

```python
self.spec_cand = []  # List of (spec_idx, cand_smiles, is_correct_label)

for spec_id, smiles in test_data:
    candidates = self.candidates[smiles]
    labels = [cand == smiles for cand in candidates]  # Binary labels

    for j, (cand, label) in enumerate(zip(candidates, labels)):
        self.spec_cand.append((spec_idx, cand, label))
```

**Example Expansion:**
```
Original: 1 spectrum → 150 candidates
Expanded: 150 items in spec_cand list
```

**`__getitem__(i)` for ExpandedRetrievalDataset:**
```python
{
    'SpecBinnerLog': Tensor[1005],
    'cand': DGLGraph,                    # candidate molecule (not ground truth)
    'cand_smiles': 'COC1=CC2=...',
    'label': True/False,                 # is this the correct molecule?
    'identifier': 'MassSpecGymID0132861'
}
```

---

## 3. DataModules

### 3.1 Base: `MassSpecDataModule`
**File:** `massspecgym/data/data_module.py`

**Key Methods:**

| Method | Action |
|--------|--------|
| `prepare_data()` | Extract fold info from metadata or split file |
| `setup(stage)` | Create Subset objects for train/val/test based on fold |
| `train_dataloader()` | Return DataLoader with shuffle=True |
| `test_dataloader()` | Return DataLoader with shuffle=False |

**Splitting Logic:**
```python
split_mask = self.split.loc[self.dataset.metadata["identifier"]].values
# "train" → train_dataset, "val" → val_dataset, "test" → test_dataset
```

### 3.2 Training: `ContrastiveDataModule`
**File:** `jestr/data/data_module.py`

Wraps training/validation data in `ContrastiveDataset`:
- Groups spectra by unique molecules
- Cycles through multiple spectra for same molecule

### 3.3 Testing: `TestDataModule`
**File:** `jestr/data/data_module.py`

Simplified module using `ExpandedRetrievalDataset`:
```python
def setup(self, stage=None):
    if stage == "test":
        self.test_dataset = self.dataset  # ExpandedRetrievalDataset
```

---

## 4. Batch Creation

### 4.1 Training Batch (ContrastiveDataset.collate_fn)

**Input:** List of items from `JESTR1_MassSpecDataset`

**Output:**
```python
{
    'SpecBinnerLog': Tensor[batch_size, 1005],  # Stacked spectra
    'mol': DGLGraph,                             # Batched graphs from dgl.batch()
    'mol_n_nodes': [14, 12, 16, ...],           # Nodes per molecule
    'identifier': ['ID001', 'ID002', ...]
}
```

**Collation Process:**
```python
# 1. Standard collate for tensors
collated_batch['SpecBinnerLog'] = default_collate([item['SpecBinnerLog'] for item in batch])

# 2. Batch molecular graphs
batch_mol = [item['mol'] for item in batch]
collated_batch['mol'] = dgl.batch(batch_mol)
collated_batch['mol_n_nodes'] = [g.num_nodes() for g in batch_mol]
```

### 4.2 Test Batch

**Input:** List of items from `ExpandedRetrievalDataset`

**Output:**
```python
{
    'SpecBinnerLog': Tensor[batch_size, 1005],
    'cand': DGLGraph,                            # Batched candidate graphs
    'mol_n_nodes': [14, 12, ...],
    'cand_smiles': ['SMILES1', 'SMILES2', ...],  # For result tracking
    'label': [True, False, False, ...],          # Ground truth labels
    'identifier': ['ID001', 'ID001', 'ID002', ...] # May repeat!
}
```

**Note:** For testing, the same spectrum ID may appear multiple times (once per candidate).

---

## 5. Result Construction

### 5.1 Test Step
**File:** `jestr/models/contrastive.py` → `ContrastiveModel.test_step()`

```python
def test_step(self, batch, batch_idx):
    # Encode spectra and candidate molecules
    spec_enc = self.spec_enc_model(batch['SpecBinnerLog'])  # [B, hidden_dim]
    mol_enc = self.mol_enc_model(batch['cand'])             # [B, hidden_dim]

    # Compute similarity scores
    scores = cosine_similarity(spec_enc, mol_enc)           # [B]

    # Group by identifier (same spectrum, multiple candidates)
    id_to_ct = defaultdict(int)
    for i in batch['identifier']: id_to_ct[i] += 1

    # Split scores by spectrum
    scores = torch.split(scores, list(id_to_ct.values()))

    return {
        'identifiers': list(id_to_ct.keys()),
        'scores': scores,                    # List of Tensors
        'cand_smiles': unbatched_smiles,
        'labels': unbatched_labels
    }
```

### 5.2 Batch End Aggregation
```python
def on_test_batch_end(self, outputs, ...):
    for i, cands, scores, labels in zip(...):
        self.result_dct[i]['candidates'].extend(cands)
        self.result_dct[i]['scores'].extend(scores.cpu().tolist())
        self.result_dct[i]['labels'].extend(labels)
```

### 5.3 Final Result DataFrame
```python
def on_test_epoch_end(self):
    # Convert accumulated results to DataFrame
    self.df_test = pd.DataFrame.from_dict(self.result_dct, orient='index')

    # Compute rank (position of correct molecule)
    self.df_test['rank'] = self.df_test.apply(
        lambda row: self._compute_rank(row['scores'], row['labels']), axis=1
    )

    # Save to pickle
    self.df_test.to_pickle(self.df_test_path)
```

**Final DataFrame Structure:**
| identifier | candidates | scores | labels | rank |
|------------|------------|--------|--------|------|
| MassSpecGymID0132861 | ['SMI1', 'SMI2', ...] | [0.85, 0.23, ...] | [True, False, ...] | 1 |
| MassSpecGymID0193514 | ['SMI3', 'SMI4', ...] | [0.72, 0.91, ...] | [False, True, ...] | 2 |

---

## 6. Complete Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. DATA LOADING                                                         │
│    data.tsv ──► MassSpecDataset                                         │
│                 ├── metadata: DataFrame                                 │
│                 └── spectra: List[matchms.Spectrum]                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. TRANSFORM CONFIGURATION                                              │
│    get_spec_featurizer() ──► SpecBinnerLog(max_mz=1005, bin_width=1)   │
│    get_mol_featurizer()  ──► MolToGraph(atom_feature='full', ...)      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. TEST DATASET CREATION                                                │
│    candidates.json + JESTR1_MassSpecDataset ──► ExpandedRetrievalDataset│
│                                                 ├── Expands each        │
│                                                 │   test spectrum to    │
│                                                 │   N candidate pairs   │
│                                                 └── Stores labels       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. DATAMODULE SETUP                                                     │
│    TestDataModule(dataset=ExpandedRetrievalDataset, collate_fn=...)    │
│    ├── prepare_data(): Extract fold information                         │
│    └── setup("test"): Assign test_dataset                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. BATCH ITERATION                                                      │
│    DataLoader ──► Batches of (spec, cand_graph, label, identifier)     │
│                   collate_fn batches graphs with dgl.batch()           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. MODEL INFERENCE                                                      │
│    ContrastiveModel.test_step(batch)                                    │
│    ├── Encode spectra: spec_enc = spec_encoder(batch['spec'])          │
│    ├── Encode candidates: mol_enc = mol_encoder(batch['cand'])         │
│    └── Score: cosine_similarity(spec_enc, mol_enc)                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. RESULT AGGREGATION                                                   │
│    on_test_batch_end(): Accumulate scores per identifier               │
│    on_test_epoch_end():                                                 │
│    ├── Build DataFrame with candidates, scores, labels                 │
│    ├── Compute rank for each spectrum                                  │
│    └── Save to pickle file                                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Key Files Reference

| Component | File Path |
|-----------|-----------|
| Base Dataset | `massspecgym/data/datasets.py` |
| JESTR Dataset | `jestr/data/datasets.py` |
| ExpandedRetrievalDataset | `jestr/data/datasets.py` |
| ContrastiveDataset | `jestr/data/datasets.py` |
| Base DataModule | `massspecgym/data/data_module.py` |
| TestDataModule | `jestr/data/data_module.py` |
| ContrastiveDataModule | `jestr/data/data_module.py` |
| Transforms (spec) | `massspecgym/data/transforms.py`, `jestr/data/transforms.py` |
| Transforms (mol) | `jestr/data/transforms.py` |
| ContrastiveModel | `jestr/models/contrastive.py` |
| Test Script | `jestr/test.py` |
| Data Utils | `jestr/utils/data.py` |

---

## 8. Example Concrete Values

### 8.1 Raw Spectrum Data
```python
# From TSV row:
identifier = "MassSpecGymID0132861"
mzs = "95.04,107.03,138.05,167.05,194.05"
intensities = "0.043,0.039,0.109,0.058,0.188"
precursor_mz = 226.07
fold = "test"
```

### 8.2 After Transform (SpecBinnerLog)
```python
# Tensor of shape [1005] with log-scaled binned intensities
tensor([0.0, 0.0, ..., 0.013, ..., 0.045, ..., 0.0])
#       ^bin 0        ^bin 95      ^bin 194    ^bin 1004
```

### 8.3 Candidate JSON Entry
```python
candidates["MassSpecGymID0132861"] = [
    "COC1=CC2=C(C=C1)N(C(=O)C(O2)O)OC",  # True positive (if matches ground truth)
    "CC(C)(C)SC(=S)Nc1ccccc1",             # Negative
    "COC(=O)Cc1c(OC)cccc1[N+](=O)[O-]",   # Negative
    # ... ~100-500 total candidates
]
```

### 8.4 Molecular Graph (DGLGraph)
```python
# For SMILES "COC1=CC2=..."
DGLGraph(
    num_nodes=14,
    num_edges=30,
    ndata={'h': Tensor[14, 127], 'm': Tensor[14, 1]},  # atom features
    edata={'e': Tensor[30, 12]}                         # bond features
)
```

### 8.5 Test Batch Example
```python
batch = {
    'SpecBinnerLog': Tensor[32, 1005],    # 32 samples
    'cand': DGLGraph(num_nodes=456),       # batched from 32 molecules
    'mol_n_nodes': [14, 12, 16, 14, ...],  # nodes per molecule
    'cand_smiles': ['SMI1', 'SMI2', ...],  # 32 SMILES
    'label': [True, False, False, ...],    # 32 labels
    'identifier': ['ID1', 'ID1', 'ID1', 'ID2', ...]  # may repeat
}
```

### 8.6 Final Result Row
```python
{
    'identifier': 'MassSpecGymID0132861',
    'candidates': ['COC1=CC2=...', 'CC(C)(C)...', ...],
    'scores': [0.923, 0.312, 0.245, ...],
    'labels': [True, False, False, ...],
    'rank': 1  # correct molecule ranked 1st
}
```
