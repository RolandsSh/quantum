from pathlib import Path
import pickle
import numpy as np
import networkx as nx

from dwave.system.samplers import DWaveSampler
from dwave.system import FixedEmbeddingComposite
from minorminer import find_embedding

def get_dwave_sampler(n_visible=100, n_hidden=100):
    dw_sampler = DWaveSampler()

    embedding_file = Path(f"embedding_{n_visible}_{n_hidden}.emb")
    if embedding_file.is_file():
        with open(embedding_file, "rb") as f:
            embedding = pickle.load(f)
    else:
        input_units = np.arange(n_visible)
        hidden_units = np.arange(n_hidden) + n_visible

        bm_edges = np.dstack(np.meshgrid(input_units, hidden_units)).reshape(-1, 2)
        bm_graph = nx.Graph(bm_edges.tolist())

        # Define the sampler that will be used to run the problem. Fix the embedding.
        print('Searching for embedding.')
        embedding = find_embedding(bm_graph, dw_sampler.to_networkx_graph())
        with open(embedding_file, "wb") as file:
            pickle.dump(embedding, file)
    dw_sampler = FixedEmbeddingComposite(dw_sampler, embedding)
    print('Embedding found and fixed.')
    return dw_sampler
