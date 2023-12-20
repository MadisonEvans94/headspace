

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class CustomLLM:
    def __init__(self, model_name="bert-base-uncased",):

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def generate_embeddings(self, inputs):
        outputs = []
        for i in inputs:
            with torch.no_grad():
                outputs.append(self.model(**i))

        return outputs
