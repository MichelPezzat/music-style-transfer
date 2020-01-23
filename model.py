import os
import tensorflow as tf
from module import *
from datetime import datetime
#from utils import *
import numpy as np
#from preprocess import *


class StarGAN(object):

    def __init__(self,
                 time_step,
                 pitch_range,
                 styles_num,
                 batchsize,
                 discriminator=discriminator,
                 generator=generator_resnet,
                 classifier=domain_classifier,
                 mode='train',
                 log_dir='./log'):
        self.time_step = time_step
        self.batchsize = batchsize

        self.input_shape = [None, time_step, pitch_range, 3]
        self.label_shape = [None, styles_num]
        self.styles_num = styles_num

        self.mode = mode
        self.log_dir = log_dir

        self.discriminator = discriminator
        self.generator = generator_resnet
        self.classifier = classifier
        self.criterionGAN = mae_criterion

        self.build_model()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            print(self.log_dir)
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
            self.generator_summaries, self.discriminator_summaries, self.domain_classifier_summaries = self.summary()
            

    def build_model(self):
        # Placeholders for real training samples
        self.input_real = tf.placeholder(tf.float32, self.input_shape, name='input_real')
        self.target_real = tf.placeholder(tf.float32, self.input_shape, name='target_real')

        self.source_label = tf.placeholder(tf.float32, self.label_shape, name='source_label')
        self.target_label = tf.placeholder(tf.float32, self.label_shape, name='target_label')

        #self. input_mixed = tf.placeholder(tf.float32, self.input_shape, name='input_mixed')
        #self. mixed_label = tf.placeholder(tf.float32, self.label_shape, name='mixed_label') 

        self.gaussian_noise = tf.placeholder(tf.float32, self.input_shape, name='gaussian_noise')

        self.generated_forward = self.generator(self.input_real, self.target_label, reuse=False, name='generator')
        self.generated_back = self.generator(self.generated_forward, self.source_label, reuse=True, name='generator')

        #Cycle loss
        self.cycle_loss = abs_criterion(self.input_real,self.generated_back)


        #Identity loss
        self.identity_map = self.generator(self.input_real, self.source_label, reuse=True, name='generator')
        self.identity_loss = abs_criterion(self.input_real, self.identity_map)

        self.discrimination_real = self.discriminator(self.target_real + self.gaussian_noise, self.target_label, reuse=False, name='discriminator')

        #combine discriminator and generator
        self.discirmination = self.discriminator(self.generated_forward + self.gaussian_noise, self.target_label, reuse=True, name='discriminator')

        self.generator_loss = self.criterionGAN(self.discirmination,tf.ones_like(self.discirmination))
        # Discriminator adversial loss

        self.discirmination_fake = self.discriminator(self.generated_forward + self.gaussian_noise, self.target_label, reuse=True, name='discriminator')

        self.discrimination_real_loss = self.criterionGAN(self.discrimination_real,tf.ones_like(self.discrimination_real))
        self.discrimination_fake_loss = self.criterionGAN(self.discirmination_fake,tf.zeros_like(self.discirmination_fake))
        

        epsilon = tf.random_uniform((self.batchsize, 1, 1, 1), 0.0, 1.0)
        x_hat = epsilon * self.generated_forward + (1.0 - epsilon) * self.input_real

        # gradient penalty
        gradients = tf.gradients(self.discriminator(x_hat, self.target_label, reuse=True, name='discriminator'), [x_hat])
        _gradient_penalty = 10.0 * tf.square(tf.norm(gradients[0], ord=2) - 1.0)


        #self.discrimination_real_all = self.discriminator(self.input_mixed, self.mixed_label, reuse = False, name ='discriminator_all')
        #self.discrimination_fake_all = self.discriminator(self.generated_forward,self.target_label, reuse = True, name = 'discriminator_all')

        #self.d_real_loss_all = self.criterionGAN(self.discrimination_real_all,tf.ones_like(self.discrimination_real_all))
        #self.d_fake_loss_all = self.criterionGAN(self.discrimination_fake_all,tf.zeros_like(self.discrimination_fake_all))
        #self.d_all_loss = (d_real_loss_all + d_fake_loss_all) / 2





        

        #domain classify loss

        self.domain_out_real = self.classifier(self.target_real, reuse=False, name='classifier')

        self.domain_out_fake = self.classifier(self.generated_forward, reuse=True, name='classifier')

        #domain_out_xxx [batchsize, 1,1,4], need to convert label[batchsize, 3] to [batchsize, 1,1,3]
        target_label_reshape = tf.reshape(self.target_label, [-1, 1, 1, self.styles_num])

        self.domain_fake_loss = softmax_criterion(self.domain_out_fake, target_label_reshape)
        self.domain_real_loss = softmax_criterion(self.domain_out_real, target_label_reshape)

        # self.domain_loss = self.domain_fake_loss + self.domain_real_loss

        # Place holder for lambda_cycle and lambda_identity
        self.lambda_cycle = tf.placeholder(tf.float32, None, name='lambda_cycle')
        self.lambda_identity = tf.placeholder(tf.float32, None, name='lambda_identity')
        self.lambda_classifier = tf.placeholder(tf.float32, None, name='lambda_classifier')
        self.lambda_mixed = tf.placeholder(tf.float32, None, name='lambda_mixed')

        self.generator_loss_all = self.generator_loss + self.lambda_cycle * self.cycle_loss + \
                                self.lambda_identity * self.identity_loss +\
                                 self.lambda_classifier * self.domain_real_loss
        self.discrimator_loss = self.discrimination_fake_loss + self.discrimination_real_loss + \
                                             _gradient_penalty + self.domain_fake_loss 

        # Categorize variables because we have to optimize the three sets of the variables separately
        trainable_variables = tf.trainable_variables()
        self.discriminator_vars = [var for var in trainable_variables if 'discriminator' in var.name]
        self.generator_vars = [var for var in trainable_variables if 'generator' in var.name]
        self.classifier_vars = [var for var in trainable_variables if 'classifier' in var.name]
        # for var in self.discriminator_vars:
        #     print(var.name)
        # for var in self.generator_vars:
        #     print(var.name)
        # for var in self.classifier_vars:
        #     print(var.name)

        #optimizer
        self.generator_learning_rate = tf.placeholder(tf.float32, None, name='generator_learning_rate')
        self.discriminator_learning_rate = tf.placeholder(tf.float32, None, name='discriminator_learning_rate')
        self.classifier_learning_rate = tf.placeholder(tf.float32, None, name="domain_classifier_learning_rate")

        self.discriminator_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.discriminator_learning_rate, beta1=0.5).minimize(
                self.discrimator_loss, var_list=self.discriminator_vars)

        self.generator_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.generator_learning_rate, beta1=0.5).minimize(
                self.generator_loss_all, var_list=self.generator_vars)

        self.classifier_optimizer = tf.train.AdamOptimizer(learning_rate=self.classifier_learning_rate).minimize(
            self.domain_real_loss, var_list=self.classifier_vars)

        # test
        self.input_test = tf.placeholder(tf.float32, self.input_shape, name='input_test')
        self.target_label_test = tf.placeholder(tf.float32, self.label_shape, name='target_label_test')

        self.generation_test = self.generator(self.input_test, self.target_label_test, reuse=True, name='generator')

        self.input_test_binary = to_binary(self.input_test,0.5) 
        self.generation_test_binary = to_binary(self.generation_test,0.5)

    def train(self, input_source, input_target, source_label, target_label, gaussian_noise,  lambda_cycle=1.0, lambda_identity=1.0, lambda_classifier=1.0, \
    generator_learning_rate=0.0001, discriminator_learning_rate=0.0001, classifier_learning_rate=0.0001):

        generation_f, _, generator_loss, _, generator_summaries = self.sess.run(
            [self.generated_forward, self.generated_back, self.generator_loss, self.generator_optimizer, self.generator_summaries], \
            feed_dict = {self.lambda_cycle: lambda_cycle, self.lambda_identity: lambda_identity, self.lambda_classifier:lambda_classifier ,\
            self.input_real: input_source, self.target_real: input_target,\
             self.source_label:source_label, self.target_label:target_label, \
             self.generator_learning_rate: generator_learning_rate,self.gaussian_noise: gaussian_noise})

        self.writer.add_summary(generator_summaries, self.train_step)

        discriminator_loss, _, discriminator_summaries = self.sess.run(\
        [self.discrimator_loss, self.discriminator_optimizer, self.discriminator_summaries], \
            feed_dict = {self.input_real: input_source, self.target_real: input_target , self.target_label:target_label,\
            self.discriminator_learning_rate: discriminator_learning_rate, self. gaussian_noise: gaussian_noise})

        self.writer.add_summary(discriminator_summaries, self.train_step)

        domain_classifier_real_loss, _, domain_classifier_summaries = self.sess.run(\
        [self.domain_real_loss, self.classifier_optimizer, self.domain_classifier_summaries],\
        feed_dict={self.input_real: input_source, self.target_label:target_label, self.target_real:input_target, \
        self.classifier_learning_rate:classifier_learning_rate}
        )
        self.writer.add_summary(domain_classifier_summaries, self.train_step)

        self.train_step += 1

        return generator_loss, discriminator_loss, domain_classifier_real_loss

    def summary(self):
        with tf.name_scope('generator_summaries'):
            cycle_loss_summary = tf.summary.scalar('cycle_loss', self.cycle_loss)
            #identity_loss_summary = tf.summary.scalar('identity_loss', self.identity_loss)

            generator_loss_summary = tf.summary.scalar('generator_loss', self.generator_loss)
            generator_summaries = tf.summary.merge([cycle_loss_summary, generator_loss_summary])

        with tf.name_scope('discriminator_summaries'):
            discriminator_loss_summary = tf.summary.scalar('discriminator_loss', self.discrimator_loss)
            discriminator_summaries = tf.summary.merge([discriminator_loss_summary])

        with tf.name_scope('domain_classifier_summaries'):
            domain_real_loss = tf.summary.scalar('domain_real_loss', self.domain_real_loss)
            domain_fake_loss = tf.summary.scalar('domain_fake_loss', self.domain_fake_loss)
            domain_classifer_summaries = tf.summary.merge([domain_real_loss, domain_fake_loss])

        return generator_summaries, discriminator_summaries, domain_classifer_summaries

    def test(self, inputs, label):
        generation,input_test = self.sess.run([self.generation_test_binary,self.input_test_binary], feed_dict={self.input_test: inputs, self.target_label_test: label})

        return generation,input_test

    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))

        return os.path.join(directory, filename)

    def load(self, filepath):
        #self.saver.restore(self.sess, filepath)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(filepath))


if __name__ == '__main__':
    starganvc = StarGAN(36)