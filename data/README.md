# Data Collection

### Protein 

The three-dimensional structures of proteins were collected from the Research Collaboratory for Structural Bioinformatics Protein Data Bank [(RCSB PDB)](https://www.rcsb.org/), which includes 204826 experimentally determined 3D structures. We focused on the PDB data format. After removing the proteins that don't have descriptions or chain A, we have 143508 left. We utilized the data version last updated on May 17, 2023. 

### Protein Description Text

In Protein Data Bank, most proteins has a primary publication linked with a PubMed ID. We collected the corresponding abstract for the proteins that contain such PubMed IDs. The file `data/esm_subset/ann.json` lists the abstract text for the proteins we sampled. We put the full abstract text file on [Google Drive](https://drive.google.com/file/d/1iMgPyiIzpvXdKiNsXnRKn2YpmP92Xyub/view?usp=share_link).
Each entry has the following format:

```json
{"pdb_id": "XXXX", "caption": "abstract of the primary publication of this protein"}
```

### Encode the Protein

We use GVP-GNN and GVP-Transformer from [ESM-IF1](https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding) to encode chain A of each protein to PyTorch tensors. The folder `data/esm_subset/pt` lists the `pt` files for the proteins we sampled. The full version after compression (83G) is on [Google Drive](https://drive.google.com/file/d/1AeJW5BY5C-d8mKJjAULTax6WA4hzWS0N/view?usp=share_link).
 