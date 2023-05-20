import os
import sys

import torch
from torch.utils.data import Dataset
import json
import numpy as np
from torch.utils.data.dataloader import default_collate

import time

class ESMDataset(Dataset):
    def __init__(self, pdb_root, ann_paths, chain="C"):
        """
        protein (string): Root directory of protein (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.pdb_root = pdb_root
        self.annotation = json.load(open(ann_paths, "r"))
        self.pdb_ids = {}
        self.chain = chain
        # self.protein_encoder = protein_encoder

    def __len__(self):
        return len(self.annotation)

    # def __getitem__(self, index):
    #     start = time.time()
    #
    #     ann = self.annotation[index]
    #
    #     coords = '{}.npy'.format(ann["pdb_id"])
    #
    #     coords_path = os.path.join(self.pdb_root, coords)
    #     end_get_path = time.time()
    #     print(f"[pdb_dataset_copy]Elapsed time end_get_path: {end_get_path - start}")
    #     coords = np.load(coords_path)
    #     end_load_coords = time.time()
    #     print(f"[pdb_dataset_copy]Elapsed time end_load_coords: {end_load_coords - end_get_path}")
    #     _, encoder_out = self.protein_encoder.sample(coords, temperature=1)
    #
    #     end_encoder_sample = time.time()
    #     print(f"[pdb_dataset_copy]Elapsed time end_encoder_sample: {end_encoder_sample - end_load_coords}")
    #
    #     caption = ann["caption"]
    #
    #     return {
    #         "pdb_coords": coords,
    #         "text_input": caption,
    #         "encoder_out": encoder_out["encoder_out"][0],
    #         "pdb_id": ann["pdb_id"]
    #     }

    # def collater(self, samples):
    #     max_len_protein_dim0 = -1
    #     for pdb_json in samples:
    #         pdb_coords = pdb_json["pdb_coords"]
    #         shape_dim0 = pdb_coords.shape[0]
    #         max_len_protein_dim0 = max(max_len_protein_dim0, shape_dim0)
    #         # print(f"max_len_protein_dim0: {max_len_protein_dim0}")
    #     for pdb_json in samples:
    #         pdb_coords = pdb_json["pdb_coords"]
    #         shape_dim0 = pdb_coords.shape[0]
    #         pad1 = ((0, max_len_protein_dim0 - shape_dim0), (0, 0), (0, 0))
    #         arr1_padded = np.pad(pdb_coords, pad1, mode='constant', )
    #         pdb_json["pdb_coords"] = arr1_padded
    #     return default_collate(samples)

    def __getitem__(self, index):
        start = time.time()

        ann = self.annotation[index]

        protein_embedding = '{}.pt'.format(ann["pdb_id"])

        protein_embedding_path = os.path.join(self.pdb_root, protein_embedding)
        end_get_path = time.time()
        # print(f"[pdb_dataset_copy]Elapsed time end_get_path: {end_get_path - start}")
        # this might cause cuda reinit issue
        protein_embedding = torch.load(protein_embedding_path, map_location=torch.device('cpu'))
        protein_embedding.requires_grad = False
        end_load_coords = time.time()
        # print(f"[pdb_dataset_copy]Elapsed time end_load_coords: {end_load_coords - end_get_path}")
        # _, encoder_out = self.protein_encoder.sample(coords, temperature=1)

        end_encoder_sample = time.time()
        # print(f"[pdb_dataset_copy]Elapsed time end_encoder_sample: {end_encoder_sample - end_load_coords}")

        caption = ann["caption"]

        return {
            "text_input": caption,
            "encoder_out": protein_embedding,
            "chain": self.chain,
            "pdb_id": ann["pdb_id"]
        }

    def collater(self, samples):
        max_len_protein_dim0 = -1
        for pdb_json in samples:
            pdb_embeddings = pdb_json["encoder_out"]
            shape_dim0 = pdb_embeddings.shape[0]
            max_len_protein_dim0 = max(max_len_protein_dim0, shape_dim0)
            # print(f"max_len_protein_dim0: {max_len_protein_dim0}")
        for pdb_json in samples:
            pdb_embeddings = pdb_json["encoder_out"]
            shape_dim0 = pdb_embeddings.shape[0]
            pad1 = ((0, max_len_protein_dim0 - shape_dim0), (0, 0), (0, 0))
            arr1_padded = np.pad(pdb_embeddings, pad1, mode='constant', )
            pdb_json["pdb_coords"] = arr1_padded
        return default_collate(samples)

