import torch
import os.path
from torchtext import data, datasets

context, answer, span = data.Field(), data.Field(), data.Field()
train, dev, test = datasets.SQUAD.splits(context, answer, span)

print("building context vocab")
context.build_vocab(train, dev)

vector_cache = "data/cache/vectors"
d_embed=300
word_vectors='glove.42B'
data_cache='.data_cache'

if os.path.isfile(vector_cache):
    context.vocab.vectors = torch.load(vector_cache)
else:
    context.vocab.load_vectors(wv_dir=data_cache, wv_type=word_vectors, wv_dim=d_embed)
    os.makedirs(os.path.dirname(vector_cache), exist_ok=True)
    torch.save(context.vocab.vectors, vector_cache)
