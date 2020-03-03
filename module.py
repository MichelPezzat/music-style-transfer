from __future__ import division
import tensorflow as tf
from ops import *
#from utils import *


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def softmax_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def discriminator_midinet(image, options, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(batch_norm(conv2d(image, options.df_dim, name='d_h0_conv'), name='d_h0_conv_bn'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(batch_norm(conv2d(h0, options.df_dim, name='d_h1_conv'), name='d_h1_conv_bn'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(batch_norm(conv2d(h1, options.df_dim, name='d_h2_conv'), name='d_h2_conv_bn'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = linear(tf.reshape(h2, [options.batch_size, -1]), options.df_dim * 16, scope='d_h3_linear')
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = linear(h3, options.output_c_dim, scope='d_h4_linear')
        return tf.nn.sigmoid(h4), h4, h0


def generator_midinet(image, options, reuse=False, name='generator'):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # h0 = lrelu(batch_norm(conv2d(image, options.df_dim, name='g_h0_conv'), name='g_h0_conv_bn'))
        # h1 = lrelu(batch_norm(conv2d(h0, options.df_dim * 2, name='g_h1_conv'), name='g_h1_conv_bn'))
        # h2 = lrelu(batch_norm(conv2d(h1, options.df_dim * 4, name='g_h2_conv'), name='g_h2_conv_bn'))
        # h3 = lrelu(batch_norm(conv2d(h2, options.df_dim * 8, name='g_h3_conv'), name='g_h3_conv_bn'))
        # h4 = lrelu(batch_norm(conv2d(h3, options.df_dim * 16, name='g_h4_conv'), name='g_h4_conv_bn'))
        h0 = tf.nn.relu(batch_norm(linear(image, options.df_dim * 16, 'g_h0_lin'), name='g_h0_lin_bn'))
        h1 = tf.nn.relu(batch_norm(linear(h0, options.df_dim * 8, 'g_h1_lin'), name='g_h1_lin_bn'))
        h1 = tf.reshape(h1, [options.batch_size, 2, 1, options.gf_dim * 4])
        h5 = tf.nn.relu(batch_norm(deconv2d(h1, options.df_dim * 2, [4, 1], [4, 1], name='g_h5_conv'), name='g_h5_conv_bn'))
        h6 = tf.nn.relu(batch_norm(deconv2d(h5, options.df_dim * 2, [4, 1], [4, 1], name='g_h6_conv'), name='g_h6_conv_bn'))
        h7 = tf.nn.relu(batch_norm(deconv2d(h6, options.df_dim * 2, [4, 1], [4, 1], name='g_h7_conv'), name='g_h7_conv_bn'))
        h8 = tf.nn.tanh(batch_norm(deconv2d(h7, options.output_c_dim, [1, 64], [1, 64], name='g_h8_conv'), name='g_h8_conv_bn'))
        # h9 = tf.nn.relu(batch_norm(deconv2d(h8, options.df_dim, name='g_h9_conv'), name='g_h9_conv_bn'))
        # h10 = tf.nn.sigmoid(batch_norm(deconv2d(h9, options.output_c_dim, name='g_h10_conv'), name='g_h10_conv_bn'))

        return h8


def discriminator_musegan_bar(input, reuse=False, name='discriminator_bar'):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        ## conv
        h0 = lrelu(conv2d(input, 128, [1, 12], [1, 12], padding='VALID', name='d_h0_conv'))
        h1 = lrelu(conv2d(h0, 128, [1, 7], [1, 7], padding='VALID', name='d_h1_conv'))
        h2 = lrelu(conv2d(h1, 128, [2, 1], [2, 1], padding='VALID', name='d_h2_conv'))
        h3 = lrelu(conv2d(h2, 128, [2, 1], [2, 1], padding='VALID', name='d_h3_conv'))
        h4 = lrelu(conv2d(h3, 256, [4, 1], [2, 1], padding='VALID', name='d_h4_conv'))
        h5 = lrelu(conv2d(h4, 512, [3, 1], [2, 1], padding='VALID', name='d_h5_conv'))

        ## linear
        h6 = tf.reshape(h5, [-1, np.product([s.value for s in h5.get_shape()[1:]])])
        h6 = lrelu(linear(h6, 1024, scope='d_h6_linear'))
        h7 = linear(h6, 1, scope='d_h7_linear')
        return h5, h7


def discriminator_musegan_phase(input, reuse=False, name='discriminator_phase'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        ## conv
        h0 = lrelu(conv2d(tf.expand_dims(input, axis=2), 512, [2, 1], [1, 1], padding='VALID', name='d_h0_conv'))
        h1 = lrelu(conv2d(h0, 128, [3, 1], [3, 1], padding='VALID', name='d_h1_conv'))

        ## linear
        h2 = tf.reshape(h1, [-1, np.product([s.value for s in h1.get_shape()[1:]])])
        h2 = lrelu(linear(h2, 1024, scope='d_h2_linear'))
        h3 = linear(h2, 1, scope='d_h3_linear')
        return h3


def generator_musegan_bar(input, output_dim, reuse=False, name='generator_bar'):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = tf.reshape(input, tf.stack([-1, 1, 1, input.get_shape()[1]]))
        h0 = relu(batch_norm(deconv2d(h0, 1024, [1, 1], [1, 1], padding='VALID', name='g_h0_deconv'),
                             name='g_h0_deconv_bn'))

        h1 = tf.reshape(h0, [-1, 2, 1, 512])
        h1 = relu(batch_norm(deconv2d(h1, 512, [2, 1], [2, 1], padding='VALID', name='g_h1_deconv'),
                             name='g_h1_deconv_bn'))

        h2 = relu(batch_norm(deconv2d(h1, 256, [2, 1], [2, 1], padding='VALID', name='g_h2_deconv'),
                             name='g_h2_deconv_bn'))

        h3 = relu(batch_norm(deconv2d(h2, 256, [2, 1], [2, 1], padding='VALID', name='g_h3_deconv'),
                             name='g_h3_deconv_bn'))

        h4 = relu(batch_norm(deconv2d(h3, 128, [2, 1], [2, 1], padding='VALID', name='g_h4_deconv'),
                             name='g_h4_deconv_bn'))

        h5 = relu(batch_norm(deconv2d(h4, 128, [3, 1], [3, 1], padding='VALID', name='g_h5_deconv'),
                             name='g_h5_deconv_bn'))

        h6 = relu(batch_norm(deconv2d(h5, 64, [1, 7], [1, 1], padding='VALID', name='g_h6_deconv'),
                             name='g_h6_deconv_bn'))

        h7 = deconv2d(h6, output_dim, [1, 12], [1, 12], padding='VALID', name='g_h7_deconv')

        return tf.nn.tanh(h7)


def generator_musegan_phase(input, output_dim, reuse=False, name='generator_phase'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = tf.reshape(input, tf.stack([-1, 1, 1, input.get_shape()[1]]))
        h0 = relu(batch_norm(deconv2d(h0, 1024, [2, 1], [2, 1], padding='VALID', name='g_h1_deconv'),
                             name='g_h1_deconv_bn'))
        h1 = relu(batch_norm(deconv2d(h0, output_dim, [3, 1], [1, 1], padding='VALID', name='g_h2_deconv'),
                             name='g_h2_deconv_bn'))
        h1 = tf.transpose(tf.squeeze(h1, axis=2), [0, 2, 1])

        return h1


def discriminator_idnet(image, style_id, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        c_cast = tf.cast(tf.reshape(style_id, [-1, 1, 1, style_id.shape[-1]]), tf.float32)
        c = tf.tile(c_cast, [1, image.shape[1], image.shape[2], 1])

        concated = tf.concat([image, c], axis=-1)

        # h0 = lrelu(conv2d(image, options.df_dim, ks=[12, 12], s=[12, 12], name='d_h0_conv'))
        # h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*4, ks=[4, 1], s=[4, 1], name='d_h1_conv'), 'd_bn1'))
        # h4 = conv2d(h1, 1, s=1, name='d_h3_pred')

        # # input is (64 x 84 x self.df_dim)
        h0 = lrelu(conv2d(concated, 32, ks=[1, 12], s=[1, 12], name='d_h0_conv'))
        c1 = tf.tile(c_cast, [1, h0.shape[1], h0.shape[2], 1])
        h0_concat = tf.concat([h0, c1], axis=-1)
        # # h0 is (64 x 7 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0_concat, 64, ks=[2, 1], s=[2, 1], name='d_h1_conv'), 'd_bn1'))
        c2 = tf.tile(c_cast, [1, h1.shape[1], h1.shape[2], 1])
        h1_concat = tf.concat([h1, c2], axis=-1)
        # # h1 is (32 x 7 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1_concat, 128, ks=[2, 1], s=[2, 1], name='d_h2_conv'), 'd_bn2'))
        c3 = tf.tile(c_cast, [1, h2.shape[1], h2.shape[2], 1])
        h2_concat = tf.concat([h2, c3], axis=-1)
        # # h2 is (16x 7 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2_concat, 256, ks=[2, 1], s=[2, 1], name='d_h3_conv'), 'd_bn3'))
        c4 = tf.tile(c_cast, [1, h3.shape[1], h3.shape[2], 1])
        h3_concat = tf.concat([h3, c4], axis=-1)
        # # h3 is (8 x 7 x self.df_dim*8)
        h4 = conv2d(h3_concat, 1, s=1, name='d_h3_pred')
        # # h4 is (8 x 7 x 1)

        #h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # (32, 42, 64)
        #h1 = lrelu(instance_norm(conv2d(h0, options.df_dim * 4, name='d_h1_conv'), 'd_bn1'))
        # (16, 21, 256)
        #h4 = conv2d(h1, 1, s=1, name='d_h3_pred')
        # (16, 21, 1)
        return h4

def discriminator(image, c, df_dim = 64,reuse=False, name="discriminator"):



    with tf.variable_scope(name):

        # image is 256 x 256 x input_c_dim

        if reuse:

            tf.get_variable_scope().reuse_variables()

        else:

            assert tf.get_variable_scope().reuse is False



        # h0 = lrelu(conv2d(image, options.df_dim, ks=[12, 12], s=[12, 12], name='d_h0_conv'))

        # h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*4, ks=[4, 1], s=[4, 1], name='d_h1_conv'), 'd_bn1'))

        # h4 = conv2d(h1, 1, s=1, name='d_h3_pred')



        # # input is (64 x 84 x self.df_dim)

        # h0 = lrelu(conv2d(image, options.df_dim, ks=[1, 12], s=[1, 12], name='d_h0_conv'))

        # # h0 is (64 x 7 x self.df_dim)

        # h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, ks=[2, 1], s=[2, 1], name='d_h1_conv'), 'd_bn1'))

        # # h1 is (32 x 7 x self.df_dim*2)

        # h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, ks=[2, 1], s=[2, 1], name='d_h2_conv'), 'd_bn2'))

        # # h2 is (16x 7 x self.df_dim*4)

        # h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, ks=[2, 1], s=[2, 1], name='d_h3_conv'), 'd_bn3'))

        # # h3 is (8 x 7 x self.df_dim*8)

        # h4 = conv2d(h3, 1, s=1, name='d_h3_pred')

        # # h4 is (8 x 7 x 1)
        c = tf.cast(tf.reshape(c, shape=[-1, 1, 1, c.shape[-1]]), tf.float32)

        c = tf.tile(c, [1, image.shape[1], image.shape[2], 1])

        image = tf.concat([image, c], axis=-1)



        h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))

        # (32, 42, 64)

        h1 = lrelu(instance_norm(conv2d(h0, df_dim * 4, name='d_h1_conv'), 'd_bn1'))

        # (16, 21, 256)

        h4 = conv2d(h1, 1, s=1, name='d_h3_pred')

        # (16, 21, 1)

        return h4




def generator_resnet(image, c, gf_dim = 64, reuse=False, name="generator"):



    with tf.variable_scope(name):

        # image is 256 x 256 x input_c_dim

        if reuse:

            tf.get_variable_scope().reuse_variables()

        else:

            assert tf.get_variable_scope().reuse is False



        def residule_block(x, dim, ks=3, s=1, name='res'):

            # e.g, x is (# of images * 128 * 128 * 3)

            p = int((ks - 1) / 2)

            # For ks = 3, p = 1

            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")

            # After first padding, (# of images * 130 * 130 * 3)

            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')

            # After first conv2d, (# of images * 128 * 128 * 3)

            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")

            # After second padding, (# of images * 130 * 130 * 3)

            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')

            # After second conv2d, (# of images * 128 * 128 * 3)

            return relu(y + x)



        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/

        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,

        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        

        c = tf.cast(tf.reshape(c, shape=[-1, 1, 1, c.shape[-1]]), tf.float32)

        c = tf.tile(c, [1, image.shape[1], image.shape[2], 1])

        image = tf.concat([image, c], axis=-1)



        # Original image is (# of images * 256 * 256 * 3)

        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        # c0 is (# of images * 262 * 262 * 3)

        c1 = relu(instance_norm(conv2d(c0, gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))

        # c1 is (# of images * 256 * 256 * 64)

        c2 = relu(instance_norm(conv2d(c1, gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))

        # c2 is (# of images * 128 * 128 * 128)

        c3 = relu(instance_norm(conv2d(c2, gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))

        # c3 is (# of images * 64 * 64 * 256)



        # c4 = relu(instance_norm(conv2d(c3, options.gf_dim*8, 3, 3, name='g_e4_c'), 'g_e4_bn'))

        # c5 = relu(instance_norm(conv2d(c4, options.gf_dim*16, 3, [4, 1], name='g_e5_c'), 'g_e5_bn'))



        # define G network with 9 resnet blocks

        r1 = residule_block(c3, gf_dim*4, name='g_r1')

        # r1 is (# of images * 64 * 64 * 256)

        r2 = residule_block(r1, gf_dim*4, name='g_r2')

        # r2 is (# of images * 64 * 64 * 256)

        r3 = residule_block(r2, gf_dim*4, name='g_r3')

        # r3 is (# of images * 64 * 64 * 256)

        r4 = residule_block(r3, gf_dim*4, name='g_r4')

        # r4 is (# of images * 64 * 64 * 256)

        r5 = residule_block(r4, gf_dim*4, name='g_r5')

        # r5 is (# of images * 64 * 64 * 256)

        r6 = residule_block(r5, gf_dim*4, name='g_r6')

        # r6 is (# of images * 64 * 64 * 256)

        r7 = residule_block(r6, gf_dim*4, name='g_r7')

        # r7 is (# of images * 64 * 64 * 256)

        r8 = residule_block(r7, gf_dim*4, name='g_r8')

        # r8 is (# of images * 64 * 64 * 256)

        r9 = residule_block(r8, gf_dim*4, name='g_r9')

        # r9 is (# of images * 64 * 64 * 256)

        r10 = residule_block(r9, gf_dim*4, name='g_r10')



        # d4 = relu(instance_norm(deconv2d(r9, options.gf_dim*8, 3, [4, 1], name='g_d4_dc'), 'g_d4_bn'))

        # d5 = relu(instance_norm(deconv2d(d4, options.gf_dim*4, 3, 3, name='g_d5_dc'), 'g_d5_bn'))



        d1 = relu(instance_norm(deconv2d(r10, gf_dim*2, 3, 2, name='g_d1_dc'), 'g_d1_bn'))

        # d1 is (# of images * 128 * 128 * 128)

        d2 = relu(instance_norm(deconv2d(d1, gf_dim, 3, 2, name='g_d2_dc'), 'g_d2_bn'))

        # d2 is (# of images * 256 * 256 * 64)

        d3 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        # After padding, (# of images * 262 * 262 * 64)

        pred = tf.nn.sigmoid(conv2d(d3, 3, 7, 1, padding='VALID', name='g_pred_c'))

        # Output image is (# of images * 256 * 256 * 3)



        return pred




def domain_classifier(image, reuse=False, name="classifier"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        # # input is 384, 84, 1
        # h0 = lrelu(conv2d(image, options.df_dim, [12, 12], [12, 12], name='d_h0_conv'))
        # # h0 is (32 x 7 x self.df_dim)
        # h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, [2, 1], [2, 1], name='d_h1_conv'), 'd_bn1'))
        # # h1 is (16 x 7 x self.df_dim*2)
        # h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, [2, 1], [2, 1], name='d_h2_conv'), 'd_bn2'))
        # # h2 is (8 x 7 x self.df_dim*4)
        # h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, [8, 1], [8, 1], name='d_h3_conv'), 'd_bn3'))
        # # h3 is (1 x 7 x self.df_dim*8)
        # h4 = conv2d(h3, 2, [1, 7], [1, 7], name='d_h3_pred')
        # # h4 is (1 x 1 x 2)

        # input is 64, 84, 1
        h0 = lrelu(conv2d(image, 64, [1, 12], [1, 12], name='d_h0_conv'))
        # h0 is (64 x 7 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, 128, [4, 1], [4, 1], name='d_h1_conv'), 'd_bn1'))
        # h1 is (16 x 7 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, 256, [2, 1], [2, 1], name='d_h2_conv'), 'd_bn2'))
        # h2 is (8 x 7 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, 512, [8, 1], [8, 1], name='d_h3_conv'), 'd_bn3'))
        # h3 is (1 x 7 x self.df_dim*8)
        h4 = conv2d(h3, 4, [1, 7], [1, 7], name='d_h3_pred')
        # h4 is (1 x 1 x 4)
        return tf.reshape(h4, [-1, 4])  # batch_size * 4







def generator(x_init, c, channel = 64,reuse=False, name="generator"):
        
        c = tf.cast(tf.reshape(c, shape=[-1, 1, 1, c.shape[-1]]), tf.float32)
        c = tf.tile(c, [1, x_init.shape[1], x_init.shape[2], 1])
        x = tf.concat([x_init, c], axis=-1)

        with tf.variable_scope(name, reuse=reuse) :
            x = conv(x, channel, kernel=7, stride=1, pad=3, use_bias=False, scope='conv')
            x = instance_norm(x, name='ins_norm')
            x = relu(x)

            # Down-Sampling
            for i in range(2) :
                x = conv(x, channel*2, kernel=4, stride=2, pad=1, use_bias=False, scope='conv_'+str(i))
                x = instance_norm(x, name='down_ins_norm_'+str(i))
                x = relu(x)

                channel = channel * 2

            # Bottleneck
            for i in range(10):
                x = resblock(x, channel, use_bias=False, scope='resblock_' + str(i))

            # Up-Sampling
            for i in range(2) :
                x = deconv(x, channel//2, kernel=4, stride=2, use_bias=False, scope='deconv_'+str(i))
                x = instance_norm(x, name='up_ins_norm'+str(i))
                x = relu(x)

                channel = channel // 2


            x = conv(x, channels=3, kernel=7, stride=1, pad=3, use_bias=False, scope='G_logit')
            x = tanh(x)

            return x

def discriminator_c(x_init, reuse=False, name="discriminator"):
        with tf.variable_scope(scope, reuse=reuse) :
            channel = 64
            x = conv(x_init, channel, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_0')
            x = lrelu(x, 0.01)

            for i in range(1, 6):
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_' + str(i))
                x = lrelu(x, 0.01)

                channel = channel * 2

            #c_kernel = int(self.img_size / np.power(2, 6))

            logit = conv(x, channels=1, kernel=3, stride=1, pad=1, use_bias=False, scope='D_logit')
            c = conv(x, channels=4, kernel=1, stride=1, use_bias=False, scope='D_label')
            c = tf.reshape(c, shape=[-1, 4])

            return logit, c

def generator_gatedcnn(inputs, speaker_id=None, reuse=False, name='generator_gatedcnn'):
    #input shape [batchsize, h, w, c]
    #speaker_id [batchsize, one_hot_vector]
    #one_hot_vector：[0,1,0,0]
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        #downsample
        d1 = downsample2d_block(inputs, filters=64, kernel_size=[7, 7], strides=[1, 1], padding=[3, 3], name_prefix='down_1')
        print(f'd1: {d1.shape.as_list()}')

        d2 = downsample2d_block(d1, filters=128, kernel_size=[3, 3], strides=[2, 2], padding=[1, 1], name_prefix='down_2')
        print(f'd2: {d2.shape.as_list()}')

        d3 = downsample2d_block(d2, filters=256, kernel_size=[3, 3], strides=[2, 2], padding=[1, 1], name_prefix='down_3')
        print(f'd3: {d3.shape.as_list()}')

        #d4 = downsample2d_block(d3, filters=64, kernel_size=[3, 5], strides=[1, 1], padding=[1, 2], name_prefix='down_4')
        #print(f'd4: {d4.shape.as_list()}')
        #d5 = downsample2d_block(d4, filters=5, kernel_size=[14, 5], strides=[14, 1], padding=[1, 2], name_prefix='down_5')
        #print(f'd5.shape :{d5.shape.as_list()}')

        for i in range(10):
                d3 = residual2d_block(d3, filters = 256,  name_prefix='resblock_' + str(i))

        #upsample
        speaker_id = tf.convert_to_tensor(speaker_id, dtype=tf.float32)
        c_cast = tf.cast(tf.reshape(speaker_id, [-1, 1, 1, speaker_id.shape.dims[-1].value]), tf.float32)
        c = tf.tile(c_cast, [1, d3.shape.dims[1].value, d3.shape.dims[2].value, 1])
        print(c.shape.as_list())
        concated = tf.concat([d3, c], axis=-1)
        # print(concated.shape.as_list())

        #u1 = upsample2d_block(concated, 64, kernel_size=[14, 5], strides=[14, 1], name_prefix='gen_up_u1')
        #print(f'u1.shape :{u1.shape.as_list()}')

        #c1 = tf.tile(c_cast, [1, u1.shape.dims[1].value, u1.shape.dims[2].value, 1])
        #print(f'c1 shape: {c1.shape}')
        #u1_concat = tf.concat([u1, c1], axis=-1)
        #print(f'u1_concat.shape :{u1_concat.shape.as_list()}')





        #u2 = upsample2d_block(concated, 128, [3, 5], [1, 1], name_prefix='gen_up_u2')
        #print(f'u2.shape :{u2.shape.as_list()}')
        #c2 = tf.tile(c_cast, [1, u2.shape[1], u2.shape[2], 1])
        #u2_concat = tf.concat([u2, c2], axis=-1)

        u3 = upsample2d_block(concated, 128, [3, 3], [2, 2], name_prefix='gen_up_u3')
        print(f'u3.shape :{u3.shape.as_list()}')
        #c3 = tf.tile(c_cast, [1, u3.shape[1], u3.shape[2], 1])
        #u3_concat = tf.concat([u3, c3], axis=-1)

        u4 = upsample2d_block(u3, 64, [3, 3], [2, 2], name_prefix='gen_up_u4')
        print(f'u4.shape :{u4.shape.as_list()}')
        #c4 = tf.tile(c_cast, [1, u4.shape[1], u4.shape[2], 1])
        #u4_concat = tf.concat([u4, c4], axis=-1)
        #print(f'u4_concat.shape :{u4_concat.shape.as_list()}')

        u5 = tf.layers.Conv2DTranspose(filters=3, kernel_size=[7, 7], strides=[1, 1], padding='same', name='generator_last_deconv')(u4)
        print(f'u5.shape :{u5.shape.as_list()}')

        return u5



def discriminator_gatedcnn(inputs, speaker_id, reuse=False, name='discriminator'):

    # inputs has shape [batch_size, height,width, channels]

    with tf.variable_scope(name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False
        #convert data type to float32
        c_cast = tf.cast(tf.reshape(speaker_id, [-1, 1, 1, speaker_id.shape[-1]]), tf.float32)
        c = tf.tile(c_cast, [1, inputs.shape[1], inputs.shape[2], 1])

        concated = tf.concat([inputs, c], axis=-1)

        # Downsample
        d1 = downsample2d_block(
            inputs=concated, filters=64, kernel_size=[4, 4], strides=[2, 2], padding=[1, 1], name_prefix='downsample2d_dis_block1_')
        #c1 = tf.tile(c_cast, [1, d1.shape[1], d1.shape[2], 1])
        #d1_concat = tf.concat([d1, c1], axis=-1)

        d2 = downsample2d_block(
            inputs=d1, filters=256, kernel_size=[4, 4], strides=[2, 2], padding=[1, 1], name_prefix='downsample2d_dis_block2_')
        #c2 = tf.tile(c_cast, [1, d2.shape[1], d2.shape[2], 1])
        #d2_concat = tf.concat([d2, c2], axis=-1)

        #d3 = downsample2d_block(
         #   inputs=d2_concat, filters=32, kernel_size=[3, 8], strides=[1, 2], padding=[1, 3], name_prefix='downsample2d_dis_block3_')
        #c3 = tf.tile(c_cast, [1, d3.shape[1], d3.shape[2], 1])
        #d3_concat = tf.concat([d3, c3], axis=-1)

        #d4 = downsample2d_block(
         #   inputs=d3_concat, filters=32, kernel_size=[3, 6], strides=[1, 2], padding=[1, 2], name_prefix='downsample2d_diss_block4_')
        #c4 = tf.tile(c_cast, [1, d4.shape[1], d4.shape[2], 1])
        #d4_concat = tf.concat([d4, c4], axis=-1)

        c1 = conv2d_layer(d2, filters=1, kernel_size=[1, 1], strides=[1, 1], padding=[1, 1], name='discriminator-last-conv')

        c1_red = tf.reduce_mean(c1, keepdims=True)

        return c1_red

def domain_classifier_b(inputs, reuse=False, name='classifier'):

    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        #   add slice input shape [batchsize, 8, 512, 1]
        #get one slice
        one_slice = inputs[:, 0:8, :, :]

        d1 = tf.layers.conv2d(one_slice, 8, kernel_size=[4, 4], padding='same', name=name + '_conv2d01')
        d1_p = tf.layers.max_pooling2d(d1, [2, 2], strides=[2, 2], name=name + 'p1')
        print(f'domain_classifier_d1: {d1.shape}')
        print(f'domain_classifier_d1_p: {d1_p.shape}')

        d2 = tf.layers.conv2d(d1_p, 16, [4, 4], padding='same', name=name + '_conv2d02')
        d2_p = tf.layers.max_pooling2d(d2, [2, 2], strides=[2, 2], name=name + 'p2')
        print(f'domain_classifier_d12: {d2.shape}')
        print(f'domain_classifier_d2_p: {d2_p.shape}')

        d3 = tf.layers.conv2d(d2_p, 32, [4, 4], padding='same', name=name + '_conv2d03')
        d3_p = tf.layers.max_pooling2d(d3, [2, 2], strides=[2, 2], name=name + 'p3')
        print(f'domain_classifier_d3: {d3.shape}')
        print(f'domain_classifier_d3_p: {d3_p.shape}')

        d4 = tf.layers.conv2d(d3_p, 16, [3, 4], padding='same', name=name + '_conv2d04')
        d4_p = tf.layers.max_pooling2d(d4, [1, 2], strides=[1, 2], name=name + 'p4')
        print(f'domain_classifier_d4: {d4.shape}')
        print(f'domain_classifier_d4_p: {d4_p.shape}')

        d5 = tf.layers.conv2d(d4_p, 4, [1, 4], padding='same', name=name + '_conv2d05')
        d5_p = tf.layers.max_pooling2d(d5, [1, 2], strides=[1, 2], name=name + 'p5')
        print(f'domain_classifier_d5: {d5.shape}')
        print(f'domain_classifier_d5_p: {d5_p.shape}')

        p = tf.keras.layers.GlobalAveragePooling2D()(d5_p)

        o_r = tf.reshape(p, [-1, 1, 1, p.shape.dims[1].value])
        print(f'classifier_output: {o_r.shape}')

        return o_r