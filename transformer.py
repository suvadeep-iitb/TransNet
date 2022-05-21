import tensorflow as tf


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, demb, **kwargs):
        super().__init__(**kwargs)

        self.inv_freq = 1 / (10000 ** (tf.range(0, demb, 2.0) / demb))

    def call(self, pos_seq, bsz=None):
        sinusoid_inp = tf.einsum("i,j->ij", pos_seq, self.inv_freq)
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)

        if bsz is not None:
            return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(tf.keras.layers.Layer):
    def __init__(self, d_model, d_inner, dropout, kernel_initializer,
                 **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.layer_1 = tf.keras.layers.Dense(
            d_inner, kernel_initializer=kernel_initializer, activation=tf.nn.relu, name='layer_1'
        )
        self.drop_1 = tf.keras.layers.Dropout(dropout, name='drop_1')
        self.layer_2 = tf.keras.layers.Dense(d_model, kernel_initializer=kernel_initializer, name='layer_2')
        self.drop_2 = tf.keras.layers.Dropout(dropout, name='drop_2')


    def call(self, inp, training=False):
        core_out = inp
        core_out = self.layer_1(core_out)
        core_out = self.drop_1(core_out, training=training)
        core_out = self.layer_2(core_out)
        core_out = self.drop_2(core_out, training=training)

        output = [core_out + inp]
        return output


class RelativeMultiHeadAttn(tf.keras.layers.Layer):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt,
        kernel_initializer,
        r_r_bias=None,
        r_w_bias=None,
        smooth_pos_emb=True,
        untie_pos_emb=True,
        clamp_len=-1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.kernel_initializer = kernel_initializer
        self.smooth_pos_emb = smooth_pos_emb
        self.untie_pos_emb = untie_pos_emb
        self.clamp_len = clamp_len

        self.qkv_net = tf.keras.layers.Dense(
            3 * n_head * d_head, kernel_initializer=kernel_initializer, use_bias=False, name="qkv"
        )

        if self.smooth_pos_emb:
            self.r_net = tf.keras.layers.Dense(
                self.n_head * self.d_head, kernel_initializer=kernel_initializer, use_bias=False, name="r"
            )
        elif self.untie_pos_emb:
            self.pos_emb = tf.keras.layers.Embedding(
                2*self.clamp_len+1, d_model, name='pos_emb'
            )

        self.drop_r = tf.keras.layers.Dropout(dropout)
        self.drop = tf.keras.layers.Dropout(dropout)
        self.dropatt = tf.keras.layers.Dropout(dropatt)
        self.o_net = tf.keras.layers.Dense(
            d_model, kernel_initializer=kernel_initializer, use_bias=False, name="o"
        )

        self.scale = 1 / (d_head ** 0.5)

        if r_r_bias is not None and r_w_bias is not None:  # Biases are shared
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias
        else:
            self.r_r_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_r_bias"
            )
            self.r_w_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_w_bias"
            )

    def _rel_shift(self, x):
        x_size = shape_list(x)

        x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
        x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
        x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        x = tf.reshape(x, x_size)

        return x

    def call(self, inputs, training=False):
        w, r = inputs
        qlen, rlen, bsz = shape_list(w)[0], shape_list(r)[0], shape_list(w)[1]

        w_heads = self.qkv_net(w)

        if not self.smooth_pos_emb and self.untie_pos_emb:
            r = self.pos_emb(r)
        r_drop = self.drop_r(r, training=training)

        if self.smooth_pos_emb:
            r_head_k = self.r_net(r_drop)
        else:
            r_head_k = r_drop

        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, axis=-1)
        w_head_q = w_head_q[-qlen:]

        klen = shape_list(w_head_k)[0]

        w_head_q = tf.reshape(w_head_q, (qlen, bsz, self.n_head, self.d_head))
        w_head_k = tf.reshape(w_head_k, (klen, bsz, self.n_head, self.d_head))
        w_head_v = tf.reshape(w_head_v, (klen, bsz, self.n_head, self.d_head))

        r_head_k = tf.reshape(r_head_k, (rlen, self.n_head, self.d_head))

        rw_head_q = w_head_q + self.r_w_bias
        rr_head_q = w_head_q + self.r_r_bias

        AC = tf.einsum("ibnd,jbnd->ijbn", rw_head_q, w_head_k)
        BD = tf.einsum("ibnd,jnd->ijbn", rr_head_q, r_head_k)
        BD = self._rel_shift(BD)
        BD = BD[:, :klen, :, :]

        attn_score = AC + BD
        attn_score = attn_score * self.scale

        attn_prob = tf.nn.softmax(attn_score, axis=1)
        attn_prob = self.dropatt(attn_prob, training=training)

        attn_vec = tf.einsum("ijbn,jbnd->ibnd", attn_prob, w_head_v)
        size_t = shape_list(attn_vec)
        attn_vec = tf.reshape(attn_vec, (size_t[0], size_t[1], self.n_head * self.d_head))

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out, training=training)

        outputs = [w + attn_out, attn_prob, AC, BD]

        return outputs


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        initializer,
        r_w_bias=None,
        r_r_bias=None,
        smooth_pos_emb=True,
        untie_pos_emb=True,
        clamp_len=-1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.d_inner = d_inner
        self.dropout = dropout
        self.dropatt = dropatt
        self.initializer = initializer
        self.smooth_pos_emb = smooth_pos_emb
        self.untie_pos_emb = untie_pos_emb
        self.clamp_len = clamp_len

        self.xltran_attn = RelativeMultiHeadAttn(
            n_head=self.n_head,
            d_model=self.d_model,
            d_head=self.d_head,
            dropout=self.dropout,
            dropatt=self.dropatt,
            kernel_initializer=self.initializer,
            r_w_bias=r_w_bias,
            r_r_bias=r_r_bias,
            smooth_pos_emb=self.smooth_pos_emb,
            untie_pos_emb=self.untie_pos_emb,
            clamp_len=self.clamp_len,
            name="xltran_attn",
        )
        self.pos_ff = PositionwiseFF(
            d_model=self.d_model,
            d_inner=self.d_inner,
            dropout=self.dropout,
            kernel_initializer=self.initializer,
            name="pos_ff",
        )

    def call(self, inputs, training=False):
        inp, r = inputs
        attn_outputs = self.xltran_attn([inp, r], training=training)
        ff_output = self.pos_ff(attn_outputs[0], training=training)

        outputs = [ff_output[0]] + attn_outputs[1:]

        return outputs



class Transformer(tf.keras.Model):
    def __init__(self, n_layer, d_model, n_head, d_head, d_inner, dropout, dropatt, 
                 n_classes, conv_kernel_size, pool_size, initializer, clamp_len=-1, 
                 untie_r=False, smooth_pos_emb=True, untie_pos_emb=True, output_attn=False):

        super(Transformer, self).__init__()

        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner

        self.dropout = dropout 
        self.dropatt = dropatt 

        self.n_classes = n_classes

        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size

        self.clamp_len = clamp_len
        self.untie_r = untie_r
        self.smooth_pos_emb = smooth_pos_emb
        self.untie_pos_emb = untie_pos_emb

        self.output_attn = output_attn

        self.initializer = initializer

        self.conv1 = tf.keras.layers.Conv1D(self.d_model, self.conv_kernel_size)
        self.relu1 = tf.keras.layers.ReLU()

        self.pool1 = tf.keras.layers.AveragePooling1D(self.pool_size, self.pool_size)

        if self.smooth_pos_emb:
            self.pos_emb = PositionalEmbedding(d_model)
        else:
            assert(self.clamp_len > 0)
            if not self.untie_pos_emb:
                self.pos_emb = tf.keras.layers.Embedding(
                    2*self.clamp_len+1, d_model, name='pos_emb'
                )
            else:
                self.pos_emb = None

        if not self.untie_r:
            self.r_w_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_w_bias"
            )
            self.r_r_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_r_bias"
            )

        self.tran_layers = []
        for i in range(self.n_layer):
            self.tran_layers.append(
                TransformerLayer(
                    n_head=self.n_head,
                    d_model=self.d_model,
                    d_head=self.d_head,
                    d_inner=self.d_inner,
                    dropout=self.dropout,
                    dropatt=self.dropatt,
                    initializer=self.initializer,
                    r_w_bias=None if self.untie_r else self.r_w_bias,
                    r_r_bias=None if self.untie_r else self.r_r_bias,
                    smooth_pos_emb=self.smooth_pos_emb,
                    untie_pos_emb=self.untie_pos_emb,
                    clamp_len=self.clamp_len,
                    name='layers_._{}'.format(i)
                )
            )

        self.out_dropout = tf.keras.layers.Dropout(dropout, name='out_drop')
        self.fc_output = tf.keras.layers.Dense(self.n_classes)

    def call(self, inp, training=False):
        # convert the input dimension from [bsz, len] to [bsz, len, 1]
        inp = tf.expand_dims(inp, axis=-1)

        # apply a single layer convolution and then perform pooling to reduce len
        inp = self.conv1(inp)
        inp = self.relu1(inp)

        inp = self.pool1(inp)

        # the rest of the code uses shapes [len, bsz, features] so we transpose 
        # here from shape [bsz, len, dimension] to shape [len, bsz, features]
        inp = tf.transpose(inp, perm=(1, 0, 2))

        slen = shape_list(inp)[0]
        pos_seq = tf.range(slen - 1, -slen, -1.0)
        if self.clamp_len > 0:
            pos_seq = tf.minimum(pos_seq, self.clamp_len)
            pos_seq = tf.maximum(pos_seq, -self.clamp_len)

        if self.smooth_pos_emb:
            pos_emb = self.pos_emb(pos_seq)
        else:
            pos_seq = pos_seq + tf.abs(tf.reduce_min(pos_seq))
            pos_emb = pos_seq if self.untie_pos_emb else self.pos_emb(pos_seq)

        core_out = inp
        out_list = []
        for i, layer in enumerate(self.tran_layers):
            all_out = layer([core_out, pos_emb], training=training)
            core_out = all_out[0]
            out_list.append(all_out[1:])
        core_out = self.out_dropout(core_out, training=training)

        # take the evarage across the first (len) dimension to get the final representation
        output = tf.reduce_mean(core_out, axis=0)

        # ge the final scores for all classes
        scores = self.fc_output(output)

        if self.output_attn:
            for i in range(len(out_list)):
                for j in range(len(out_list[i])):
                    out_list[i][j] = tf.transpose(out_list[i][j], perm=[2, 3, 0, 1])
            return [scores] + out_list
        else:
            return [scores]
        

