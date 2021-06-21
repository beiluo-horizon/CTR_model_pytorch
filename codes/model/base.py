import itertools
from collections import OrderedDict

import torch.nn as nn


class EmbeddingMixin:
    def build_embeddings(
            self, default_embedding_size, fixed_embedding_size=False):
        embeddings = OrderedDict()
        embedding_sizes = OrderedDict()

        for feature in itertools.chain(
                self.features.category_features,
                self.features.sequence_features):
            if feature.embedding_name not in embeddings:
                embedding_size = default_embedding_size
                if not fixed_embedding_size:
                    embedding_size = (feature.embedding_size
                                      if feature.embedding_size
                                      else default_embedding_size)

                #创建两个有序字典
                #分别保存了embedding层和embedding_size
                #每个embedding层的输入是特征的数量
                embeddings[feature.embedding_name] = nn.Embedding(
                    feature.dimension(), embedding_size, padding_idx=0)
                embedding_sizes[feature.embedding_name] = embedding_size

                #torch.add_module 可以在init函数添加一层
                self.add_module(
                    f"embedding:{feature.embedding_name}",
                    embeddings[feature.embedding_name])

            if feature.name != feature.embedding_name:
                embeddings[feature.name] = embeddings[feature.embedding_name]
                embedding_sizes[feature.name] = (
                    embedding_sizes[feature.embedding_name])
                if feature.embedding_size and (
                        feature.embedding_size !=
                        embedding_sizes[feature.name]):
                    raise RuntimeWarning(
                        f"embedding_size of {feature.name} should be "
                        f"the same with {feature.embedding_name}")

        return (embeddings, embedding_sizes)
