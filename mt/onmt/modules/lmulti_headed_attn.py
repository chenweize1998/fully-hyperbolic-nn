""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn

from onmt.modules.hyper_nets import LorentzLinear


class LorentzMultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """
    def __init__(self,
                 head_count,
                 model_dim,
                 manifold,
                 dropout=0.1,
                 max_relative_positions=0):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(LorentzMultiHeadedAttention, self).__init__()
        self.manifold = manifold
        self.head_count = head_count

        self.linear_keys = LorentzLinear(model_dim,
                                         head_count * self.dim_per_head,
                                         dropout=dropout,
                                         manifold=manifold,
                                         head_num=head_count)
        self.linear_values = LorentzLinear(model_dim,
                                           head_count * self.dim_per_head,
                                           dropout=dropout,
                                           manifold=manifold,
                                           head_num=head_count)
        self.linear_query = LorentzLinear(model_dim,
                                          head_count * self.dim_per_head,
                                          dropout=dropout,
                                          manifold=manifold,
                                          head_num=head_count)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor([math.sqrt(model_dim)]))
        self.bias = nn.Parameter(torch.zeros(()))

        self.max_relative_positions = max_relative_positions

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head)

    def forward(self,
                key,
                value,
                query,
                mask=None,
                layer_cache=None,
                attn_type=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * Attention vector in heads ``(batch, head, query_len, key_len)``.
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """Projection."""
            if len(x.size()) == 3:
                x = x.view(batch_size, -1, head_count, dim_per_head)
            return x.transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2)  #.contiguous()
            # .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                query, key, value = self.linear_query(query),\
                    self.linear_keys(query),\
                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat((layer_cache["self_keys"], key), dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat((layer_cache["self_values"], value),
                                      dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif attn_type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                        self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"],\
                        layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)
        query = shape(query)
        key_len = key.size(2)
        query_len = query.size(2)

        attn = (2 +
                2 * self.manifold.cinner(query, key)) / self.scale + self.bias
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            attn = attn.masked_fill(mask, -1e18)
        attn = self.softmax(attn)

        context = self.manifold.mid_point(value, attn)

        context = unshape(context)

        # output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return multi-head attn
        attns = attn \
            .view(batch_size, head_count,
                  query_len, key_len)

        return context, attns

    def update_dropout(self, dropout):
        self.dropout.p = dropout
