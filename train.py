import torch
import os.path
from model import BIDAF
from torchtext import data, datasets


def train(config):
    context, answer, span = data.Field(), data.Field(), data.Field()
    train, dev, test = datasets.SQUAD.splits(context, answer, span, config=config)
    print("building context vocab")
    context.build_vocab(train, dev)
    train.config.word_vocab_size = len(context.vocab)

    if os.path.isfile(config.vector_cache):
        context.vocab.vectors = torch.load(config.vector_cache)
    else:
        context.vocab.load_vectors(wv_dir=config.data_cache, wv_type=config.word_vectors, wv_dim=config.d_embed)
        os.makedirs(os.path.dirname(config.vector_cache), exist_ok=True)
        torch.save(context.vocab.vectors, config.vector_cache)

    model = BIDAF(train.config)
