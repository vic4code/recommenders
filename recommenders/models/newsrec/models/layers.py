# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from tkinter import X
import tensorflow.compat.v1.keras as keras
from tensorflow.compat.v1.linalg import einsum
from tensorflow.compat.v1.keras import layers, initializers
from tensorflow.compat.v1.keras import backend as K


class AttLayer2(layers.Layer):
    """Soft alignment attention implement.

    Attributes:
        dim (int): attention hidden dim
    """

    def __init__(self, dim=200, seed=0, **kwargs):
        """Initialization steps for AttLayer2.

        Args:
            dim (int): attention hidden dim
        """

        self.dim = dim
        self.seed = seed
        super(AttLayer2, self).__init__(**kwargs)

    def build(self, input_shape):
        """Initialization for variables in AttLayer2
        There are there variables in AttLayer2, i.e. W, b and q.

        Args:
            input_shape (object): shape of input tensor.
        """

        assert len(input_shape) == 3
        dim = self.dim
        self.W = self.add_weight(
            name="W",
            shape=(int(input_shape[-1]), dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(dim,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
        )
        self.q = self.add_weight(
            name="q",
            shape=(dim, 1),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        super(AttLayer2, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, inputs, mask=None, **kwargs):
        """Core implemention of soft attention

        Args:
            inputs (object): input tensor.

        Returns:
            object: weighted sum of input tensors.
        """

        attention = K.tanh(K.dot(inputs, self.W) + self.b)
        attention = K.dot(attention, self.q)

        attention = K.squeeze(attention, axis=2)

        if mask is None:
            attention = K.exp(attention)
        else:
            attention = K.exp(attention) * K.cast(mask, dtype="float32")

        attention_weight = attention / (
            K.sum(attention, axis=-1, keepdims=True) + K.epsilon()
        )

        attention_weight = K.expand_dims(attention_weight)
        weighted_input = inputs * attention_weight
        return K.sum(weighted_input, axis=1)

    def compute_mask(self, input, input_mask=None):
        """Compte output mask value

        Args:
            input (object): input tensor.
            input_mask: input mask

        Returns:
            object: output mask.
        """
        return None

    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor

        Args:
            input_shape (tuple): shape of input tensor.

        Returns:
            tuple: shape of output tensor.
        """
        return input_shape[0], input_shape[-1]


class SelfAttention(layers.Layer):
    """Multi-head self attention implement.

    Args:
        multiheads (int): The number of heads.
        head_dim (object): Dimention of each head.
        mask_right (boolean): whether to mask right words.

    Returns:
        object: Weighted sum after attention.
    """

    def __init__(self, multiheads, head_dim, seed=0, mask_right=False, **kwargs):
        """Initialization steps for AttLayer2.

        Args:
            multiheads (int): The number of heads.
            head_dim (object): Dimension of each head.
            mask_right (boolean): Whether to mask right words.
        """

        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        self.seed = seed
        super(SelfAttention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor.

        Returns:
            tuple: output shape tuple.
        """

        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def build(self, input_shape):
        """Initialization for variables in SelfAttention.
        There are three variables in SelfAttention, i.e. WQ, WK ans WV.
        WQ is used for linear transformation of query.
        WK is used for linear transformation of key.
        WV is used for linear transformation of value.

        Args:
            input_shape (object): shape of input tensor.
        """

        self.WQ = self.add_weight(
            name="WQ",
            shape=(int(input_shape[0][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.WK = self.add_weight(
            name="WK",
            shape=(int(input_shape[1][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.WV = self.add_weight(
            name="WV",
            shape=(int(input_shape[2][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        super(SelfAttention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode="add"):
        """Mask operation used in multi-head self attention

        Args:
            seq_len (object): sequence length of inputs.
            mode (str): mode of mask.

        Returns:
            object: tensors after masking.
        """

        if seq_len is None:
            return inputs
        else:
            mask = K.one_hot(indices=seq_len[:, 0], num_classes=K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, axis=1)

            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)

            if mode == "mul":
                return inputs * mask
            elif mode == "add":
                return inputs - (1 - mask) * 1e12

    def call(self, QKVs):
        """Core logic of multi-head self attention.

        Args:
            QKVs (list): inputs of multi-head self attention i.e. query, key and value.

        Returns:
            object: ouput tensors.
        """
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs
            Q_len, V_len = None, None
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(
            Q_seq, shape=(-1, K.shape(Q_seq)[1], self.multiheads, self.head_dim)
        )
        Q_seq = K.permute_dimensions(Q_seq, pattern=(0, 2, 1, 3))

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(
            K_seq, shape=(-1, K.shape(K_seq)[1], self.multiheads, self.head_dim)
        )
        K_seq = K.permute_dimensions(K_seq, pattern=(0, 2, 1, 3))

        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(
            V_seq, shape=(-1, K.shape(V_seq)[1], self.multiheads, self.head_dim)
        )
        V_seq = K.permute_dimensions(V_seq, pattern=(0, 2, 1, 3))

        A = einsum("abij, abkj -> abik", Q_seq, K_seq) / K.sqrt(
            K.cast(self.head_dim, dtype="float32")
        )
        A = K.permute_dimensions(
            A, pattern=(0, 3, 2, 1)
        )  # A.shape=[batch_size,K_sequence_length,Q_sequence_length,self.multiheads]

        A = self.Mask(A, V_len, "add")
        A = K.permute_dimensions(A, pattern=(0, 3, 2, 1))

        if self.mask_right:
            ones = K.ones_like(A[:1, :1])
            lower_triangular = K.tf.matrix_band_part(ones, num_lower=-1, num_upper=0)
            mask = (ones - lower_triangular) * 1e12
            A = A - mask
        A = K.softmax(A)

        O_seq = einsum("abij, abjk -> abik", A, V_seq)
        O_seq = K.permute_dimensions(O_seq, pattern=(0, 2, 1, 3))

        O_seq = K.reshape(O_seq, shape=(-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, "mul")
        return O_seq

    def get_config(self):
        """add multiheads, multiheads and mask_right into layer config.

        Returns:
            dict: config of SelfAttention layer.
        """
        config = super(SelfAttention, self).get_config()
        config.update(
            {
                "multiheads": self.multiheads,
                "head_dim": self.head_dim,
                "mask_right": self.mask_right,
            }
        )
        return config


def PersonalizedAttentivePooling(dim1, dim2, dim3, seed=0):
    """Soft alignment attention implement.

    Attributes:
        dim1 (int): first dimention of value shape.
        dim2 (int): second dimention of value shape.
        dim3 (int): shape of query

    Returns:
        object: weighted summary of inputs value.
    """
    vecs_input = keras.Input(shape=(dim1, dim2), dtype="float32")
    query_input = keras.Input(shape=(dim3,), dtype="float32")

    user_vecs = layers.Dropout(0.2)(vecs_input)
    user_att = layers.Dense(
        dim3,
        activation="tanh",
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed),
        bias_initializer=keras.initializers.Zeros(),
    )(user_vecs)
    user_att2 = layers.Dot(axes=-1)([query_input, user_att])
    user_att2 = layers.Activation("softmax")(user_att2)
    user_vec = layers.Dot((1, 1))([user_vecs, user_att2])

    model = keras.Model([vecs_input, query_input], user_vec)
    return model


class ComputeMasking(layers.Layer):
    """Compute if inputs contains zero value.

    Returns:
        bool tensor: True for values not equal to zero.
    """

    def __init__(self, **kwargs):
        super(ComputeMasking, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mask = K.not_equal(inputs, 0)
        return K.cast(mask, K.floatx())

    def compute_output_shape(self, input_shape):
        return input_shape


class OverwriteMasking(layers.Layer):
    """Set values at spasific positions to zero.

    Args:
        inputs (list): value tensor and mask tensor.

    Returns:
        object: tensor after setting values to zero.
    """

    def __init__(self, **kwargs):
        super(OverwriteMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        super(OverwriteMasking, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs[0] * K.expand_dims(inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class AttentionPooling(layers.Layer):

    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.att1 = layers.Dense(hidden_size)
        self.att2 = layers.Dense(1)
    
    def call(self, inputs, attention_mask):
        """
        Args:
            inputs: inputs (object): input tensor. (batch, seq_len, dim)
        Return:
            outputs: (batch, seq_len, dim)
        """
        batch_size = inputs.shape[0]
        e = self.att1(inputs)
        e = K.tanh(e)
        alpha = self.att2(e)
        alpha = K.exp(alpha)
        if attention_mask is not None:
            alpha = alpha * K.expand_dims(attention_mask, axis=2)
        alpha = alpha / (K.sum(alpha, axis=1, keepdims=True) + 1e-8)
        inputs = K.permute_dimensions(inputs, (0, 2, 1))
        inputs = K.batch_dot(inputs, alpha, axes=[2, 1])
        inputs = K.reshape(inputs, (batch_size, -1))

        return inputs


class FastSelfAttention(layers.Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        self.now_input_shape=None
        super(FastSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.now_input_shape = input_shape
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True) 
        self.Wq = self.add_weight(name='Wq', 
                                  shape=(self.output_dim,self.nb_head),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Wk = self.add_weight(name='Wk', 
                                  shape=(self.output_dim,self.nb_head),
                                  initializer='glorot_uniform',
                                  trainable=True)
        
        self.WP = self.add_weight(name='WP', 
                                  shape=(self.output_dim,self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        
        
        super(FastSelfAttention, self).build(input_shape)
        
    def call(self, inputs):
        """
        Args:
            inputs: inputs (object): input tensor. (batch, seq_len)
        Return:
            outputs: (batch)
        """

        if len(inputs) == 2:
            Q_seq, K_seq = inputs
        elif len(inputs) == 4:
            Q_seq, K_seq, Q_mask, K_mask = inputs #different mask lengths, reserved for cross attention

        Q_seq = K.dot(Q_seq, self.WQ)        
        Q_seq_reshape = K.reshape(Q_seq, (-1, self.now_input_shape[0][1], self.nb_head * self.size_per_head))

        Q_att = K.permute_dimensions(K.dot(Q_seq_reshape, self.Wq), (0, 2, 1)) / self.size_per_head ** 0.5

        if len(inputs)  == 4:
            Q_att = Q_att - (1 - K.expand_dims(Q_mask, axis=1)) * 1e8

        Q_att = K.softmax(Q_att)
        Q_seq = K.reshape(Q_seq, (-1, self.now_input_shape[0][1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, self.now_input_shape[1][1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))

        Q_att = layers.Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=3), self.size_per_head, axis=3))(Q_att)
        global_q = K.sum(layers.multiply([Q_att, Q_seq]), axis=2)
        
        global_q_repeat = layers.Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=2), self.now_input_shape[1][1], axis=2))(global_q)

        QK_interaction = layers.multiply([K_seq, global_q_repeat])
        QK_interaction_reshape = K.reshape(QK_interaction, (-1, self.now_input_shape[0][1], self.nb_head * self.size_per_head))
        K_att = K.permute_dimensions(K.dot(QK_interaction_reshape, self.Wk), (0, 2, 1)) / self.size_per_head ** 0.5
        
        if len(inputs) == 4:
            K_att = K_att - (1 - K.expand_dims(K_mask, axis=1)) * 1e8
            
        K_att = K.softmax(K_att)
        K_att = layers.Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=3), self.size_per_head, axis=3))(K_att)

        global_k = K.sum(layers.multiply([K_att, QK_interaction]), axis=2)
        global_k_repeat = layers.Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=2), self.now_input_shape[0][1], axis=2))(global_k)
        
        #Q=V
        QKQ_interaction = layers.multiply([global_k_repeat, Q_seq])
        QKQ_interaction = K.permute_dimensions(QKQ_interaction, (0, 2, 1, 3))
        QKQ_interaction = K.reshape(QKQ_interaction, (-1, self.now_input_shape[0][1], self.nb_head * self.size_per_head))
        QKQ_interaction = K.dot(QKQ_interaction, self.WP)
        QKQ_interaction = K.reshape(QKQ_interaction, (-1, self.now_input_shape[0][1], self.nb_head, self.size_per_head))
        QKQ_interaction = K.permute_dimensions(QKQ_interaction, (0, 2, 1, 3))
        QKQ_interaction = QKQ_interaction + Q_seq
        QKQ_interaction = K.permute_dimensions(QKQ_interaction, (0, 2, 1, 3))
        QKQ_interaction = K.reshape(QKQ_interaction, (-1, self.now_input_shape[0][1], self.nb_head * self.size_per_head))

        return QKQ_interaction
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


class FastAttention(layers.Layer):

    def __init__(self, nb_head, size_per_head, hidden_size, hidden_dropout_prob, layer_norm_eps):
        super(FastAttention, self).__init__()
        self.selfattn = FastSelfAttention(nb_head, size_per_head)
        self.linear_module = keras.Sequential()
        self.linear_module.add(layers.Dense(hidden_size))
        self.linear_module.add(layers.Dropout(hidden_dropout_prob))
        self.LayerNorm = layers.LayerNormalization(epsilon=layer_norm_eps)

    def call(self, inputs, attention_mask):
        """
        Args:
            inputs: inputs (object): input tensor. (batch, seq_len, dim)
        Return:
            outputs: (batch, seq_len, dim)
        """
        self_output = self.selfattn([inputs, inputs, attention_mask, attention_mask])
        attention_output = self.linear_module(self_output)
        attention_output = self.LayerNorm(attention_output + inputs)

        return attention_output


class FastformerLayer(layers.Layer):

    def __init__(self, 
                 nb_head, 
                 size_per_head, 
                 hidden_size, 
                 intermediate_size, 
                 hidden_dropout_prob, 
                 layer_norm_eps):

        super(FastformerLayer, self).__init__()
        self.attention = FastAttention(nb_head, 
                                       size_per_head, 
                                       hidden_size,
                                       hidden_dropout_prob, 
                                       layer_norm_eps)

        self.intermediate = layers.Dense(intermediate_size, activation='gelu')
 
        self.linear_module = keras.Sequential()
        self.linear_module.add(layers.Dense(hidden_size))
        self.linear_module.add(layers.Dropout(hidden_dropout_prob))
      
        self.LayerNorm = layers.LayerNormalization(epsilon=layer_norm_eps)

    def call(self, inputs, attention_mask):
        attention_output = self.attention(inputs, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.linear_module(intermediate_output)
        layer_output = self.LayerNorm(layer_output + attention_output)

        return layer_output

class FastformerEncoder(layers.Layer):
    def __init__(self, 
                 nb_head, 
                 size_per_head, 
                 hidden_size, 
                 intermediate_size, 
                 hidden_dropout_prob, 
                 layer_norm_eps, 
                 num_hidden_layers, 
                 max_position_embeddings,
                 pooler_type='weightpooler',
                 pooler_count=1
                 ):
                       
        super(FastformerEncoder, self).__init__()

        self.encoders = []
        for _ in range(num_hidden_layers):
            self.encoders.append(
                                FastformerLayer(
                                    nb_head, 
                                    size_per_head, 
                                    hidden_size, 
                                    intermediate_size,
                                    hidden_dropout_prob, 
                                    layer_norm_eps
                                ) 
            )

        self.position_embeddings = layers.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = layers.LayerNormalization(epsilon=layer_norm_eps)
        self.dropout = layers.Dropout(hidden_dropout_prob)

        # support multiple different poolers with shared bert encoder.
        self.poolers = []
        if pooler_type == 'weightpooler':
            for _ in range(pooler_count):
                self.poolers.append(AttentionPooling(hidden_size))

    def call(self, 
             inputs, 
             attention_mask, 
             pooler_index=0
             ):
        #input_embs: batch_size, seq_len, emb_dim
        #attention_mask: batch_size, seq_len, emb_dim

        extended_attention_mask = K.expand_dims(attention_mask, axis=1)
        # extended_attention_mask = attention_mask
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        batch_size, seq_length, emb_dim = inputs.shape
        position_ids = K.arange(seq_length, dtype='float32')
        position_ids = K.expand_dims(position_ids, axis=0)
        position_ids = K.repeat_elements(position_ids, batch_size, axis=0)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        all_hidden_states = [embeddings]

        # print(all_hidden_states)
        print(all_hidden_states[-1].shape, extended_attention_mask.shape)

        for i, layer_module in enumerate(self.encoders):
            layer_outputs = layer_module(all_hidden_states[-1], extended_attention_mask)
            all_hidden_states.append(layer_outputs)
        assert len(self.poolers) > pooler_index
        output = self.poolers[pooler_index](all_hidden_states[-1], attention_mask)

        return output 

class Fastformer(layers.Layer):

    def __init__(self,
                 nb_head, 
                 size_per_head, 
                 hidden_size, 
                 intermediate_size, 
                 hidden_dropout_prob, 
                 layer_norm_eps, 
                 num_hidden_layers, 
                 max_position_embeddings,
                 pooler_type='weightpooler',
                 pooler_count=1
                 ):

        super(Fastformer, self).__init__()
   
        self.word_embedding = layers.Embedding(max_position_embeddings, hidden_size, mask_zero=True)
        self.fastformer_model = FastformerEncoder(nb_head, 
                                                  size_per_head, 
                                                  hidden_size, 
                                                  intermediate_size, 
                                                  hidden_dropout_prob, 
                                                  layer_norm_eps, 
                                                  num_hidden_layers, 
                                                  max_position_embeddings,
                                                  pooler_type='weightpooler',
                                                  pooler_count=1)

    def call(self, inputs, mask):
        text_vec = self.fastformer_model(inputs, mask)
        return text_vec
