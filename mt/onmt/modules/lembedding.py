""" Embeddings module """
import math
import warnings

import torch
import torch.nn as nn

from onmt.modules.util_class import Elementwise
from onmt.modules.hyper_nets import LorentzLinear
from geoopt import ManifoldParameter
from onmt.manifolds import Lorentz
from onmt.manifolds.utils import acosh


class LorentzPositionEncoding(nn.Module):
    def __init__(self, dropout, dim, manifold, max_len=5000, learnable=True):
        super(LorentzPositionEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        self.manifold = manifold
        if not learnable:
            if dim % 2 != 0:
                raise ValueError("Cannot use sin/cos positional encoding with "
                                 "odd dim (got dim={:d})".format(dim))
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                                  -(math.log(10000.0) / dim)))
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            pe = pe.unsqueeze(1)
        else:
            pe = nn.Parameter(torch.randn(max_len, 1, dim) * 0.02)
        self.pe = pe
        self.dropout = nn.Dropout(p=0.1)
        self.dim = dim
        self.point = LorentzLinear(dim, dim)

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        pe = self.pe[:emb.size(0)] if step is None else self.pe[step]
        emb = self.point(emb, pe)

        return emb


class LorentzEmbeddings(nn.Module):
    """Words embeddings for encoder/decoder.

    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.


    .. mermaid::

       graph LR
          A[Input]
          C[Feature 1 Lookup]
          A-->B[Word Lookup]
          A-->C
          A-->D[Feature N Lookup]
          B-->E[MLP/Concat]
          C-->E
          D-->E
          E-->F[Output]

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        feat_padding_idx (List[int]): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes (List[int], optional): list of size of dictionary
            of embeddings for each feature.
        position_encoding (bool): see :class:`~onmt.modules.PositionalEncoding`
        feat_merge (string): merge action for the features embeddings:
            concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
            embedding size is N^feat_dim_exponent, where N is the
            number of values the feature takes.
        feat_vec_size (int): embedding dimension for features when using
            `-feat_merge mlp`
        dropout (float): dropout probability.
    """

    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
                 position_encoding=False,
                 feat_merge="concat",
                 feat_vec_exponent=0.7,
                 feat_vec_size=-1,
                 feat_padding_idx=[],
                 feat_vocab_sizes=[],
                 dropout=0,
                 sparse=False,
                 fix_word_vecs=False):
        self._validate_args(feat_merge, feat_vocab_sizes, feat_vec_exponent,
                            feat_vec_size, feat_padding_idx)

        if feat_padding_idx is None:
            feat_padding_idx = []
        self.word_padding_idx = word_padding_idx

        word_vec_size = word_vec_size
        self.word_vec_size = word_vec_size

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else word_vec_size)

        super(LorentzEmbeddings, self).__init__()

        self.manifold = Lorentz()
        self.embedding = ManifoldParameter(self.manifold.random_normal(
            (vocab_sizes[0], emb_dims[0])), manifold=self.manifold)
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}

        self.pe = LorentzPositionEncoding(
            dropout, self.embedding_size, self.manifold, learnable=True)

    def _validate_args(self, feat_merge, feat_vocab_sizes, feat_vec_exponent,
                       feat_vec_size, feat_padding_idx):
        if feat_merge == "sum":
            # features must use word_vec_size
            if feat_vec_exponent != 0.7:
                warnings.warn("Merging with sum, but got non-default "
                              "feat_vec_exponent. It will be unused.")
            if feat_vec_size != -1:
                warnings.warn("Merging with sum, but got non-default "
                              "feat_vec_size. It will be unused.")
        elif feat_vec_size > 0:
            # features will use feat_vec_size
            if feat_vec_exponent != -1:
                warnings.warn("Not merging with sum and positive "
                              "feat_vec_size, but got non-default "
                              "feat_vec_exponent. It will be unused.")
        else:
            if feat_vec_exponent <= 0:
                raise ValueError("Using feat_vec_exponent to determine "
                                 "feature vec size, but got feat_vec_exponent "
                                 "less than or equal to 0.")
        n_feats = len(feat_vocab_sizes)
        if n_feats != len(feat_padding_idx):
            raise ValueError("Got unequal number of feat_vocab_sizes and "
                             "feat_padding_idx ({:d} != {:d})".format(
                                 n_feats, len(feat_padding_idx)))

    def load_pretrained_vectors(self, emb_file):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
        """

        if emb_file:
            pretrained = torch.load(emb_file)
            pretrained_vec_size = pretrained.size(1)
            if self.word_vec_size > pretrained_vec_size:
                self.word_lut.weight.data[:, :pretrained_vec_size] = pretrained
            elif self.word_vec_size < pretrained_vec_size:
                self.word_lut.weight.data \
                    .copy_(pretrained[:, :self.word_vec_size])
            else:
                self.word_lut.weight.data.copy_(pretrained)

    def forward(self, source, step=None):
        """Computes the embeddings for words and features.

        Args:
            source (LongTensor): index tensor ``(len, batch, nfeat)``

        Returns:
            FloatTensor: Word embeddings ``(len, batch, embedding_size)``
        """

        shape = source.shape[:-1]
        source = self.embedding.index_select(
            0, source.view(-1, )).view(shape + (-1,))
        return self.pe(source, step)

    def update_dropout(self, dropout):
        if self.position_encoding:
            self._modules['make_embedding'][1].dropout.p = dropout
        # pass
