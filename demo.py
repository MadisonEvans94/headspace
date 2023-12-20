import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.manifold import TSNE
import os
from dimensionality_reducer import DimensionalityReducer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

sample_text = [
    "Artificial intelligence is transforming our lives in profound ways.",
    "The development of self-driving cars marks a revolution in transportation.",
    "Quantum computing has the potential to solve complex problems.",
    "Blockchain technology is reshaping financial transactions and data security.",
    "Virtual reality offers new dimensions for gaming and education.",
    "The rise of 5G networks is accelerating mobile communication capabilities.",
    "Robotic automation is changing manufacturing processes.",
    "The Internet of Things connects everyday devices to the internet.",
    "Machine learning algorithms are becoming more sophisticated at data analysis.",
    "Renewable energy technologies are crucial for sustainable future.",
    "Forests play a vital role in maintaining Earth's ecosystem.",
    "Conservation efforts are essential to protect endangered wildlife.",
    "The beauty of the ocean is vast but vulnerable to pollution.",
    "Climate change is a significant challenge for global sustainability.",
    "Urban green spaces are important for environmental balance.",
    "Sustainable agriculture practices contribute to food security.",
    "Biodiversity is key to a resilient natural environment.",
    "Protecting natural habitats is critical for preserving biodiversity.",
    "The Amazon rainforest is home to a myriad of species.",
    "Renewable resources like wind and solar power are key to a clean energy future."
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


tsne = DimensionalityReducer.create_reducer(
    "tsne", perplexity=3, learning_rate=200)

reduced_embeddings = tsne.reduce_dimensions(embeddings_np)


# 2D Visualization
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

# Adding labels and title for clarity
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('2D t-SNE Projection of BERT Sentence Embeddings')

# Show the plot
plt.show()
