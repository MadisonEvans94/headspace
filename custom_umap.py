import umap
import matplotlib.pyplot as plt
import numpy as np
import torch


class CustomUMAP:
    def __init__(self, n_components=2, n_neighbors=3) -> None:
        self.n_components = n_components

        self.n_neighbors = n_neighbors

        # Initialize the UMAP object for 2D (or 3D) reduction
        self.reducer = umap.UMAP(
            n_components=self.n_components, n_neighbors=self.n_neighbors, low_memory=True)

    def to_numpy(self, embeddings):
        print("Converting embeddings to NumPy array...")

        # Averaging the token embeddings to get a single vector per sentence
        embedding_average = [torch.mean(embedding.squeeze(), dim=0)
                             for embedding in embeddings]

        # Convert the list of tensors to a NumPy array
        embedding_np = torch.stack(
            embedding_average).detach().numpy()

        print("Converted array shape:", embedding_np.shape)
        return embedding_np

    def reduce_dimensions(self, embeddings):
        print("Reducing dimensions...")

        out = self.reducer.fit_transform(embeddings)
        return out
