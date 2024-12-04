import math
import mindspore as ms
import mindspore.common.dtype as mstype
from typing import Optional
from mindspore import nn, __version__
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindformers.models.llama.llama_layer import LlamaRMSNorm
from mindformers.modules.layers import _check_input_dtype, Dropout, RotaryEmbedding
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.modules.flash_attention import FlashAttention
from tzchat_layer import tzchatLinear, tzchatFeedForward


class _MicroBatch(nn.Cell):
    def __init__(self, micro_size, input_size, axis_list):
        super(_MicroBatch, self).__init__()
        self.shape = P.Shape()
        self.micro_size = micro_size
        self.strided_slice_list = []
        for _ in range(input_size):
            self.strided_slice_list.append(P.StridedSlice())
        self.axis_list = axis_list

    def construct(self, i, *inputs):
        micro_inputs = ()
        k = 0
        for each_input in inputs:
            input_shape = self.shape(each_input)
            micro_batch_begin = i * input_shape[self.axis_list[k]] // self.micro_size
            micro_batch_end = (i + 1) * input_shape[self.axis_list[k]] // self.micro_size
            strided_slice_begin = ()
            strided_slice_strides = ()
            strided_slice_end = ()
            for j, _ in enumerate(input_shape):
                strided_slice_strides += (1,)
                if j == self.axis_list[k]:
                    strided_slice_begin += (micro_batch_begin,)
                    strided_slice_end += (micro_batch_end,)
                else:
                    strided_slice_begin += (0,)
                    strided_slice_end += (input_shape[j],)

            micro_input = self.strided_slice_list[k](each_input, strided_slice_begin,\
                strided_slice_end, strided_slice_strides)
            micro_inputs += (micro_input,)
            k += 1
        return micro_inputs


class tzchatAttentionInterleave(nn.Cell):
    def __init__(self,
                 seq_length,
                 dim: int = 512,
                 n_heads: int = 8,
                 sigma: float = 0.0048,
                 mean: float = 0.0,
                 hidden_dropout_prob: float = 1.0,
                 attention_dropout_prob: float = 1.0,
                 n_kv_heads: Optional[int] = None,
                 compute_dtype=mstype.float16,
                 softmax_compute_dtype=mstype.float32,
                 rotary_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 qkv_has_bias=False,
                 wo_has_bias=True,
                 is_dynamic=False,
                 use_rope_slice=False,
                 use_flash_attention=False,
                 parallel_config=TransformerOpParallelConfig()):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_size = dim
        self.n_head = n_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.head_dim = dim // n_heads
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_head // self.n_kv_head
        self.kv_dim = self.n_kv_head * self.head_dim
        self.qkv_has_bias = qkv_has_bias
        self.wo_has_bias = wo_has_bias
        self.dtype = compute_dtype
        self.softmax_dtype = softmax_compute_dtype
        self.is_first_iteration = True
        self.use_flash_attention = use_flash_attention

        if self.hidden_size % self.n_head != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'n_head', but got the hidden_size is {} and the n_head is {}."
                             .format(self.hidden_size, self.n_head))
        if self.n_kv_head % parallel_config.model_parallel != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'n_kv_head' must be a multiple of "
                             "'parallel_config.model_parallel', but got the n_kv_head is {} "
                             "and the parallel_config.model_parallel  is {}."
                             .format(self.n_kv_head, parallel_config.model_parallel))

        self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.head_dim), dtype=compute_dtype)

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.merger_head_transpose = P.Transpose()
        self.batch_matmul = P.BatchMatMul()
        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.mul = P.Mul()
        self.add = P.Add()
        self.softmax = P.Softmax()
        self.cast = P.Cast()
        self.cast_attn = P.Cast()
        self.tile_kv = P.Tile()
        self.split_kv = ms.ops.auto_generate.SplitWithSize()
        self.split_kv.add_prim_attr("skip_redistribution", True)
        self.apply_rotary_emb = RotaryEmbedding(self.head_dim, rotary_dtype, use_rope_slice=use_rope_slice)
        self.attention_dropout = Dropout(1 - self.attention_dropout_prob)

        self.wq = tzchatLinear(self.hidden_size,
                                 self.hidden_size,
                                 has_bias=qkv_has_bias,
                                 sigma=sigma,
                                 mean=mean,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type,
                                 skip_redistribution=is_dynamic)
        self.wk_v = tzchatLinear(self.hidden_size,
                                   self.n_kv_head * self.head_dim * 2,
                                   has_bias=qkv_has_bias,
                                   sigma=sigma,
                                   mean=mean,
                                   compute_dtype=compute_dtype,
                                   param_init_type=param_init_type,
                                   skip_redistribution=is_dynamic)
        self.wo = tzchatLinear(in_channels=self.hidden_size,
                                 out_channels=self.hidden_size,
                                 has_bias=wo_has_bias,
                                 sigma=sigma,
                                 mean=mean,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type,
                                 skip_redistribution=is_dynamic,
                                 keep_prob=1 - self.hidden_dropout_prob)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.split_kv.shard(((dp, mp, 1),))
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.transpose.shard(((dp, 1, mp, 1),))
            self.merger_head_transpose.shard(((dp, mp, 1, 1),))
            self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.mul.shard(((dp, mp, 1, 1), ()))
            self.add.shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
            self.softmax.shard(((dp, mp, 1, 1),))
            self.tile_kv.shard(((dp, mp, 1, 1),))

            self.apply_rotary_emb.shard(parallel_config)

            if self.qkv_has_bias:
                self.wq.shard(((dp, 1), (mp, 1)), ((dp, mp), (mp,)))
                self.wk_v.shard(((dp, 1), (mp, 1)), ((dp, mp), (mp,)))
            else:
                self.wq.shard(((dp, 1), (mp, 1)))
                self.wk_v.shard(((dp, 1), (mp, 1)))
            if self.wo_has_bias:
                self.wo.shard(((dp, mp), (1, mp)), ((dp, 1), (1,)))
            else:
                self.wo.shard(((dp, mp), (1, mp)))
            if parallel_config.use_seq_parallel and self.is_first_iteration:
                if self.wo_has_bias:
                    self.wo.shard(((dp, mp), (1, mp)), ((dp * mp, 1), (1,)), out_strategy_matmul=((dp * mp, 1),))
                else:
                    self.wo.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))
            if parallel_config.recompute.select_recompute and not self.use_flash_attention:
                self.apply_rotary_emb.recompute()
                self.tile_kv.recompute()
                self.batch_matmul_q_k.recompute()
                self.mul.recompute()
                self.add.recompute()
                self.cast_attn.recompute()
                self.softmax.recompute()
                self.batch_matmul.recompute()

        if self.use_flash_attention:
            self.flash_attention = FlashAttention(head_num=self.n_head,
                                                  pre_tokens=65536,
                                                  next_tokens=0,
                                                  input_layout='BNSD',
                                                  keep_prob=1. - attention_dropout_prob,
                                                  scale_value=1. / math.sqrt(self.head_dim),
                                                  sparse_mode=0,
                                                  use_attention_mask=True)
            self.flash_attention.shard(parallel_config)

    def compute_qkv(self, x):
        x = self.reshape(x, (-1, x.shape[-1]))
        query = self.cast(self.wq(x), self.dtype)  # dp, 1 -> dp, mp
        key_value = self.cast(self.wk_v(x), self.dtype)    # dp, 1 -> dp, mp
        key_value = self.reshape(key_value, (-1, self.n_kv_head, self.head_dim * 2))
        key, value = self.split_kv(key_value, (self.head_dim, self.head_dim), 2)
        key = self.reshape(key, (-1, self.n_kv_head * self.head_dim))
        value = self.reshape(value, (-1, self.n_kv_head * self.head_dim))
        return query, key, value

    def cal_attn(self, query, key, value, mask, freqs_cis):
        query = self.reshape(query, (-1, self.seq_length, self.n_head, self.head_dim))
        key = self.reshape(key, (-1, self.seq_length, self.n_kv_head, self.head_dim))
        value = self.reshape(value, (-1, self.seq_length, self.n_kv_head, self.head_dim))

        # [bs, seq/1, n_head/n_kv_head, head_dim]
        query = self.transpose(query, (0, 2, 1, 3))
        key = self.transpose(key, (0, 2, 1, 3))
        value = self.transpose(value, (0, 2, 1, 3))

        # [bs, n_head/n_kv_head, seq/1, head_dim]
        query, key = self.apply_rotary_emb(query, key, freqs_cis) # dp, mp, 1, 1
        # kv share: [bs, n_kv_head, seq, head_dim] -> [bs, n_head, seq, head_dim]
        bs, n_head, seq, head_dim = query.shape
        n_kv_head = key.shape[1]
        query = self.reshape(query, (bs, n_head, seq, head_dim))
        key = self.reshape(key, (bs, n_kv_head, seq, head_dim))
        value = self.reshape(value, (bs, n_kv_head, seq, head_dim))

        # q, k, v: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim], [bs, n_head, seq, head_dim]
        if self.use_flash_attention:
            attention = self.flash_attention(query, key, value, mask)
            attention = self._merge_heads(attention)
        else:
            key = self._repeat_kv(key, self.n_rep)
            value = self._repeat_kv(value, self.n_rep)
            attention = self._attn(query, key, value, mask)
        return attention

    def cal_output_proj(self, attention):
        output = self.wo(attention) # dp, mp -> dp, 1 / dp * mp, 1
        return output

    def _repeat_kv(self, x, rep):
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = x.shape
        x = self.reshape(x, (bs, n_kv_head, 1, seqlen * head_dim))
        x = self.tile_kv(x, (1, 1, rep, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def _merge_heads(self, x):
        # [bs, n_head, seq/1, head_dim]
        x = self.merger_head_transpose(x, (0, 2, 1, 3)) # dp,mp,1,1 -> dp,1,mp,1
        # [bs, seq/1, n_head, head_dim]
        x_shape = x.shape
        # [bs * seq/1, hidden_dim]
        new_shape = (-1, x_shape[-2] * x_shape[-1])
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, mask):
        # q, k: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim]
        score = self.batch_matmul_q_k(query, key)
        # score: [bs, n_head, seq/1, seq]
        score = self.mul(score, self.inv_norm_factor)
        score = self.add(mask, score)

        attention_probs = self.softmax(self.cast_attn(score, self.softmax_dtype))
        attention_probs = self.attention_dropout(attention_probs)
        # score, v: [bs, n_head, seq/1, seq], [bs, n_head, seq, head_dim]
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.dtype), value)
        # [bs, n_head, seq/1, head_dim]
        attention_merge = self._merge_heads(weighted_values)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        return attention_merge


class tzchatDecodeLayerInterleave(nn.Cell):
    def __init__(self,
                 seq_length,
                 layer_id,
                 dim: int = 512,
                 n_heads: int = 8,
                 num_layers: int = 32,
                 sigma: float = 0.0048,
                 mean: float = 0.0,
                 hidden_dropout_prob: float = 1.0,
                 attention_dropout_prob: float = 1.0,
                 n_kv_heads: Optional[int] = None,
                 intermediate_size: Optional[int] = None,
                 ffn_dim_multiplier: Optional[int] = None,
                 norm_eps: float = 1e-5,
                 compute_dtype=mstype.float16,
                 layernorm_compute_dtype=mstype.float32,
                 softmax_compute_dtype=mstype.float32,
                 rotary_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 res_dtype=mstype.float32,
                 qkv_has_bias=False,
                 wo_has_bias=True,
                 is_dynamic=False,
                 use_rope_slice=False,
                 use_flash_attention=False,
                 fine_grain_interleave=2,
                 parallel_config=TransformerOpParallelConfig()):

        super().__init__()
        self.seq_length = seq_length
        self.layer_id = layer_id
        self.hidden_size = dim
        self.n_head = n_heads
        self.num_layers = num_layers
        self.head_dim = self.hidden_size // self.n_head
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads

        self.dtype = compute_dtype
        self.res_dtype = res_dtype
        self.is_first_iteration = True
        self.interleave_num = fine_grain_interleave
        self.key_past = None
        self.value_past = None

        self.reshape = P.Reshape()
        self.add = P.Add()
        self.cast = P.Cast()
        self.attention_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.ffn_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.attention = tzchatAttentionInterleave(seq_length=seq_length,
                                                     dim=dim,
                                                     n_heads=n_heads,
                                                     sigma=sigma,
                                                     mean=mean,
                                                     hidden_dropout_prob=hidden_dropout_prob,
                                                     attention_dropout_prob=attention_dropout_prob,
                                                     n_kv_heads=n_kv_heads,
                                                     compute_dtype=compute_dtype,
                                                     softmax_compute_dtype=softmax_compute_dtype,
                                                     rotary_dtype=rotary_dtype,
                                                     param_init_type=param_init_type,
                                                     qkv_has_bias=qkv_has_bias,
                                                     wo_has_bias=wo_has_bias,
                                                     is_dynamic=is_dynamic,
                                                     use_rope_slice=use_rope_slice,
                                                     use_flash_attention=use_flash_attention,
                                                     parallel_config=parallel_config)
        self.feed_forward = tzchatFeedForward(dim=self.hidden_size,
                                                intermediate_size=intermediate_size,
                                                hidden_dim=4 * self.hidden_size,
                                                sigma=sigma,
                                                mean=mean,
                                                hidden_dropout_prob=hidden_dropout_prob,
                                                ffn_dim_multiplier=ffn_dim_multiplier,
                                                compute_dtype=compute_dtype,
                                                param_init_type=param_init_type,
                                                is_dynamic=is_dynamic)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.feed_forward.shard(parallel_config)
            self.add.shard(((dp, 1), (dp, 1)))
            self.attention_norm.shard((dp, 1))
            self.ffn_norm.shard((dp, 1))

        if parallel_config.use_seq_parallel and self.is_first_iteration:
            self.add.shard(((dp * mp, 1), (dp * mp, 1)))
            self.attention_norm.shard((dp * mp, 1))
            self.ffn_norm.shard((dp * mp, 1))
            self.feed_forward.w2.shard(((dp, mp), (1, mp)), ((dp * mp, 1), (1,)), out_strategy_matmul=((dp * mp, 1),))

        concat_stra1 = []
        concat_stra2 = []
        self.interleave1_inputs = nn.CellList()
        self.interleave1_inputs_ = nn.CellList()
        self.interleave2_inputs = nn.CellList()
        self.interleaved_concat1 = P.Concat(axis=0)
        self.interleaved_concat1.add_prim_attr("fine_grained_interleaved_index", self.layer_id)
        self.interleaved_concat_1 = P.Concat(axis=0)
        self.interleaved_concat2 = P.Concat(axis=0)
        if self.layer_id != self.num_layers - 2:
            self.interleaved_concat2.add_prim_attr("fine_grained_interleaved_index", 1000)

        for _ in range(self.interleave_num):
            concat_stra1.append((dp, mp))
            interleave_data1 = _MicroBatch(self.interleave_num, 1, [0])
            interleave_data1.strided_slice_list[0].add_prim_attr("skip_redistribution", True)
            interleave_data1_ = _MicroBatch(self.interleave_num, 1, [0])
            interleave_data1_.strided_slice_list[0].add_prim_attr("skip_redistribution", True)
            interleave_data2 = _MicroBatch(self.interleave_num, 2, [0, 0])
            if parallel_config.use_seq_parallel:
                if self.layer_id == self.num_layers - 2:
                    concat_stra2.append((dp, 1))
                else:
                    concat_stra2.append((dp * mp, 1))
                if self.layer_id == self.num_layers - 1:
                    interleave_data1.strided_slice_list[0].shard(((dp, 1),))
                else:
                    interleave_data1.strided_slice_list[0].shard(((dp * mp, 1),))
                interleave_data1_.strided_slice_list[0].shard(((1, 1),))
                interleave_data2.strided_slice_list[0].shard(((dp * mp, 1),))
            else:
                concat_stra2.append((dp, 1))
                interleave_data1.strided_slice_list[0].shard(((dp, 1),))
                interleave_data1_.strided_slice_list[0].shard(((1, 1),))
                interleave_data2.strided_slice_list[0].shard(((dp, 1),))
            if self.layer_id == 0 and parallel_config.use_seq_parallel:
                interleave_data2.strided_slice_list[0].shard(((dp, 1),))
                interleave_data2.strided_slice_list[0].add_prim_attr("skip_redistribution", True)
            else:
                interleave_data2.strided_slice_list[0].add_prim_attr("skip_redistribution", True)

            interleave_data2.strided_slice_list[0].add_prim_attr("fine_grained_interleaved_index", self.layer_id)
            interleave_data2.strided_slice_list[1].shard(((dp, mp),))
            interleave_data2.strided_slice_list[1].add_prim_attr("fine_grained_interleaved_index", self.layer_id)
            interleave_data2.strided_slice_list[1].add_prim_attr("skip_redistribution", True)
            self.interleave1_inputs.append(interleave_data1)
            self.interleave1_inputs_.append(interleave_data1_)
            self.interleave2_inputs.append(interleave_data2)
        concat_stra3 = tuple(concat_stra1)
        concat_stra4 = tuple(concat_stra2)
        self.interleaved_concat1.shard(concat_stra3)
        self.interleaved_concat1.add_prim_attr("skip_redistribution", True)
        self.interleaved_concat_1.shard(concat_stra3)
        self.interleaved_concat_1.add_prim_attr("skip_redistribution", True)
        self.interleaved_concat2.shard(concat_stra4)
        self.interleaved_concat2.add_prim_attr("skip_redistribution", True)

    def linear_layer1(self, x):
        input_x = self.attention_norm(x)
        query, key, value = self.attention.compute_qkv(input_x)
        return query, key, value

    def linear_layer2(self, x, attention):
        attention_output = self.attention.cal_output_proj(attention)
        ori_dtype = attention_output.dtype
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        x = self.add(self.cast(x, self.res_dtype), self.cast(attention_output, self.res_dtype))
        output_x = self.ffn_norm(x)
        mlp_logit = self.feed_forward(output_x)
        output = self.add(self.cast(x, self.res_dtype), self.cast(mlp_logit, self.res_dtype))
        output = self.cast(output, ori_dtype)
        return output

    # pylint: disable=W0613
    def construct(self, x, freqs_cis, mask=None, batch_valid_length=None, block_tables=None,
                  slot_mapping=None, prefix_keys_values=None, q_seq_lens=None):
        self._check_input(x, freqs_cis, mask)
        x = self.reshape(x, (-1, x.shape[-1]))
        # ============linear-layer1================
        if self.layer_id == 0:
            query, key, value = self.linear_layer1(x)
        else:
            query_tuple = ()
            key_tuple = ()
            value_tuple = ()
            for i in range(self.interleave_num):
                x_part, = self.interleave1_inputs[i](i, x)
                query_part, key_part, value_part = self.linear_layer1(x_part)
                query_tuple += (query_part,)
                key_tuple += (key_part,)
                value_tuple += (value_part,)
            query = self.interleaved_concat1(query_tuple)
            key = self.interleaved_concat_1(key_tuple)
            value = self.interleaved_concat_1(value_tuple)
        # ===========linear-layer1 end=============
        attention = self.attention.cal_attn(query, key, value, mask, freqs_cis)
        # ============linear-layer2================
        if self.layer_id == self.num_layers - 1:
            output = self.linear_layer2(x, attention)
        else:
            output_tuple = ()
            for i in range(self.interleave_num):
                x_part, attention_part = self.interleave2_inputs[i](i, x, attention)
                output_part = self.linear_layer2(x_part, attention_part)
                output_tuple += (output_part,)
            output = self.interleaved_concat2(output_tuple)
         # ============linear-layer2 end===========
        return output

    def _check_input(self, x, freqs_cis, mask):
        _check_input_dtype(
            x.dtype, "x", [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        freqs_cos, freqs_sin, swap_mask = freqs_cis
        _check_input_dtype(freqs_cos.dtype, "freqs_cos",
                           [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        _check_input_dtype(freqs_sin.dtype, "freqs_sin",
                           [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        if swap_mask is not None:
            _check_input_dtype(swap_mask.dtype, "swap_mask",
                               [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        if mask is not None:
            _check_input_dtype(mask.dtype, "input_mask",
                               [mstype.float32, mstype.float16, mstype.uint8, mstype.bfloat16], self.cls_name)
        return True
