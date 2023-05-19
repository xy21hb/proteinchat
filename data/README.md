# Data Collection

### Protein 

The three-dimensional structures of proteins were collected from the Research Collaboratory for Structural Bioinformatics Protein Data Bank [(RCSB PDB)](https://www.rcsb.org/), which includes 204,826 experimentally determined 3D structures. We focused on the PDB data format. We utilized the data version last updated on May 17, 2023. 

### Protein Description Text

In Protein Data Bank, most proteins has a primary publication linked with a PubMed ID. We collected the corresponding abstract for the 163,635 proteins that contain such PubMed IDs. The file `data/annotation.json` lists the abstract texts. 
Each entry has the following format:

```json
{"pdb_id": "XXXX", "caption": "abstract of the primary publication of this protein"}
```

### Encoded Protein

We use GVP-GNN and GVP-Transformer from [ESM-IF1](examples/inverse_folding) to encode PDB files to PyTorch tensors. 
