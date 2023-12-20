import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.manifold import TSNE
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
sample_text = [
    "Restoring natural habitats helps maintain ecological balance.",
    "Restoring natural habitats helps maintain earth balance.",
    "Restoring natural habitats helps maintain water balance.",
    "Restoring natural habitats helps maintain heat balance.",
    "asdf ass gdasga.",
    "asd fass gdasga.",
    "asdf assgda sga.",
    "asdf   assgdasga.",
]

inputs = [tokenizer(text, return_tensors="pt",
                    truncation=True, max_length=512) for text in sample_text]

outputs = []
for i in inputs:
    with torch.no_grad():
        outputs.append(model(**i))

# Assuming you want the last hidden states (token-level embeddings)
embeddings = [output[0] for output in outputs]

# Averaging the token embeddings to get a single vector per sentence
embedding_average = [torch.mean(embedding.squeeze(), dim=0)
                     for embedding in embeddings]

# Convert the list of tensors to a NumPy array
embeddings_np = torch.stack(
    embedding_average).detach().numpy()

# Adjust perplexity to be less than the number of samples
tsne = TSNE(n_components=2, perplexity=5)
reduced_embeddings = tsne.fit_transform(embeddings_np)


# 2D Visualization
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

# Adding labels and title for clarity
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('2D t-SNE Projection of BERT Sentence Embeddings')

# Show the plot
plt.show()
