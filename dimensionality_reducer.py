from abc import ABC, abstractmethod
import numpy as np
from sklearn.manifold import TSNE
# You would import additional dimensionality reduction libraries here, like UMAP


class DimensionalityReducer(ABC):
    """
    Abstract class for dimensionality reduction. All concrete implementations
    should follow this interface.
    """

    @abstractmethod
    def reduce_dimensions(self, data: np.ndarray):
        """
        Reduce the dimensions of the given data.

        :param data: A numpy array of the data to reduce.
        :return: The data transformed into a lower-dimensional space.
        """
        pass

    @staticmethod
    def create_reducer(type: str, **kwargs):
        """
        Factory method to create a dimensionality reducer.

        :param type: The type of reducer to create (e.g., 'tsne').
        :param kwargs: Additional keyword arguments to pass to the reducer's constructor.
        :return: An instance of a DimensionalityReducer subclass.
        """
        if type == "tsne":
            return TSNEReducer(**kwargs)
        # Add additional elif branches here for other reducer types.
        else:
            raise ValueError(f"Unknown reducer type: {type}")


class TSNEReducer(DimensionalityReducer):
    """
    t-SNE dimensionality reduction implementation.
    """

    def __init__(self, n_components=2, perplexity=30, learning_rate=200):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate

    def reduce_dimensions(self, data: np.ndarray):
        tsne = TSNE(n_components=self.n_components,
                    perplexity=self.perplexity, learning_rate=self.learning_rate)
        return tsne.fit_transform(data)
