import umap
import matplotlib.pyplot as plt
import numpy as np
import torch


class CustomUMAP:
    def __init__(self, n_components=2, random_state=42, n_neighbors=3) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.n_neighbors = n_neighbors

        # Initialize the UMAP object for 2D (or 3D) reduction
        self.reducer = umap.UMAP(n_components=self.n_components,
                                 random_state=self.random_state, n_neighbors=self.n_neighbors)

    def to_numpy(self, embeddings):

        # Averaging the token embeddings to get a single vector per sentence
        embedding_average = [torch.mean(embedding.squeeze(), dim=0)
                             for embedding in embeddings]

        # Convert the list of tensors to a NumPy array
        embedding_np = torch.stack(
            embedding_average).detach().numpy()

        return embedding_np

    def reduce(self, embeddings):

        return self.reducer.fit_transform(embeddings)
