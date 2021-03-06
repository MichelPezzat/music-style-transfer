import os
from os.path import abspath, join, dirname
import numpy as np
import argparse
import time
import librosa
import glob
#from preprocess import *
from model import *
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utility import *

TIME_STEP = 64
PITCH_RANGE = 84
STYLES_NUM = 4
MODEL_NAME = 'stargan_model'

def get_files_labels(pattern: str):
    files = glob.glob(pattern)
    names = []
    for f in files:
        t = f.rsplit('/', maxsplit=1)[1]  #'./data/jazz_classic_pop/jazz_piano_train_273.npy'
        name = t.rsplit('.', maxsplit=1)[0]
        names.append(name)

    return files, names


def train(processed_dir: str, test_wav_dir: str):
    timestr = time.strftime("%Y-%m-%d-%H-%M", time.localtime()) 

    #restore_dir = './5_2019-12-10-06-04/model/' #like '2018-10-10-14-47'

    all_styles = get_styles(processed_dir,all_styles=[])
    label_enc = LabelEncoder()
    label_enc.fit(all_styles)

    lambda_cycle = 10
    lambda_identity = 5
    lambda_classifier = 3

    sigma = 0.1

    generator_learning_rate = 0.0002
    generator_learning_rate_decay = generator_learning_rate / 20000
    discriminator_learning_rate = 0.0002
    discriminator_learning_rate_decay = discriminator_learning_rate / 20000
    domain_classifier_learning_rate = 0.0001
    domain_classifier_learning_rate_decay = domain_classifier_learning_rate / 20000
    #====================load data================#
    print('Loading Data...')

    files, names = get_files_labels(os.path.join(processed_dir, '*.npy'))
    
    assert len(files) > 0

    #normlizer = Normalizer()

    exclude_dict = {}  #key that not appear in the value list.(eg. pop:[classic**.npy,classic**.npy,jazz**.npy ... ])
    for s in all_styles:
        p = os.path.join(processed_dir, '*.npy')  #'./data/jazz_classic_pop/*.npy'
        temp = [fn for fn in glob.glob(p) if fn.rsplit('/', maxsplit=1)[1].find(s) == -1]
        exclude_dict[s] = temp

    print('Loading Data Done.')

    #====================create model=============#
    BATCHSIZE = 8
    model = StarGAN(time_step=TIME_STEP, pitch_range=PITCH_RANGE,styles_num =STYLES_NUM,batchsize = BATCHSIZE)
    #model.load(filepath)
    #====================start train==============#
    EPOCH = 101

    num_samples = len(files)
    for epoch in range(EPOCH):
        start_time_epoch = time.time()

        files_shuffled, names_shuffled = shuffle(files, names)
        data_mixed,names_mixed = shuffle(files_shuffled,names_shuffled)

        for i in range(num_samples // BATCHSIZE):
            num_iterations = num_samples // BATCHSIZE * epoch + i

            gaussian_noise = np.abs(np.random.normal(0, sigma, (BATCHSIZE, TIME_STEP,
                                                                         PITCH_RANGE, 3)))

            if num_iterations > 2500:
                
                domain_classifier_learning_rate = max(0, domain_classifier_learning_rate - domain_classifier_learning_rate_decay)
                generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
                discriminator_learning_rate = max(0, discriminator_learning_rate - discriminator_learning_rate_decay)

            #if discriminator_learning_rate == 0 or generator_learning_rate == 0:
             #   print('Early stop training.')
              #  break
            # if num_iterations > 2500:
            #     lambda_identity = 1
            #     domain_classifier_learning_rate = 0
            #     generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
              #  discriminator_learning_rate = discriminator_learning_rate + discriminator_learning_rate_decay

            if generator_learning_rate <= 0.0001:
                 generator_learning_rate = 0.0001
            if discriminator_learning_rate <= 0.0001:
                 discriminator_learning_rate = 0.0001

            start = i * BATCHSIZE
            end = (i + 1) * BATCHSIZE

            if end > num_samples:
                end = num_samples

            X, X_t, X_norm, X_m,y, y_t, y_m = [], [], [], [], [], [], []

            #get target file paths
            batchnames = names_shuffled[start:end]
            pre_targets = []
            for name in batchnames:
                name = name.split(sep='_')[0]  #pop
                t = np.random.choice(exclude_dict[name], 1)[0]
                pre_targets.append(t)
            #one batch train data
            for one_filename, one_name, one_filename_mixed, one_name_mixed,one_target in zip(files_shuffled[start:end], names_shuffled[start:end], \
                       data_mixed[start:end], names_mixed[start:end], pre_targets):

                #target name
                t = one_target.rsplit('/', maxsplit=1)[1]  #'./data/jazz_classic_pop/pop_piano_train_688.npy'
                target_style_name = t.rsplit('.', maxsplit=1)[0].split('_')[0]


                #source name
                style_name = one_name.split('_')[0]  #pop

                #shape [1,64,84,1]
                one_file = np.load(one_filename)*1.
                X.append(one_file)
                
                #source label
                temp_index = label_enc.transform([style_name])[0]
                temp_arr_s = np.zeros([
                    len(all_styles),
                ])
                temp_arr_s[temp_index] = 1
                y.append(temp_arr_s)

                #load target files and labels
                

                one_file_t = np.load(one_target)*1.
                one_file_norm = one_file_t*2.-1.
                

                #[1,84,64,1]
                
                X_t.append(one_file_t)


                X_norm.append(one_file_norm)


                


                #target label
                temp_index_t = label_enc.transform([target_style_name])[0]
                temp_arr_t = np.zeros([
                    len(all_styles),
                ])
                temp_arr_t[temp_index_t] = 1
                y_t.append(temp_arr_t)

                one_file_mixed = np.load(one_filename_mixed)*1.

                X_m.append(one_file_mixed)

                style_mixed_name = one_name_mixed.split('_')[0] 

                temp_index_m = label_enc.transform([style_mixed_name])[0]
                temp_arr_m = np.zeros([
                    len(all_styles),
                ])
                temp_arr_m[temp_index_m] = 1
                y_m.append(temp_arr_m)

               



                
            

            
             
            generator_loss, discriminator_loss, domain_classifier_loss = model.train(\
            input_source=X, input_target=X_t, input_norm =X_norm, input_mixed = X_m,source_label=y, \
            target_label=y_t, mixed_label = y_m, generator_learning_rate=generator_learning_rate,\
             discriminator_learning_rate=discriminator_learning_rate,\
            classifier_learning_rate=domain_classifier_learning_rate, \
            lambda_identity=lambda_identity, lambda_cycle=lambda_cycle,\
            lambda_classifier=lambda_classifier, gaussian_noise = gaussian_noise\
            )

            print('Iteration: {:07d}, Generator Learning Rate: {:.7f}, Discriminator Learning Rate: {:.7f},Generator Loss : {:.3f}, Discriminator Loss : {:.3f}, domain_classifier_loss: {:.3f}'\
            .format(num_iterations, generator_learning_rate, discriminator_learning_rate, generator_loss, \
            discriminator_loss, domain_classifier_loss))

        #=======================test model==========================

        if epoch % 1 == 0 and epoch != 0:
            print('============test model============')
            #out put path
            file_path = os.path.join('./out', f'{epoch}_{timestr}')
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            tempfiles = []
            for one_style in all_styles:
                p = os.path.join(test_wav_dir, f'{one_style}/*.npy')
                npys = glob.glob(p)
                npys.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
                tempfiles.append(list(zip(npys[:BATCHSIZE]))) 
                 #'./data/test/pop/pop_piano_test.npy'

            for one_style_batch in tempfiles:
                _, style, name = one_style_batch[0][0].rsplit('/', maxsplit=2)
                sample_images = [np.load(one_style_batch[0])*1. for one_style_batch in one_style_batch]
                sample_images = np.array(sample_images).astype(np.float32)
                targets = [fn for fn in all_styles if style!=fn]
                for target in targets:


                     

                     #target label 1->2, 2->3, 3->0, 0->1
                     source_test_sample_label = np.zeros([len(one_style_batch),len(all_styles)])
                     target_test_sample_label = np.zeros([len(one_style_batch),len(all_styles)])
                     temp_index_s = label_enc.transform([style])[0]
                     temp_index_t = label_enc.transform([target])[0]

                
                     for i in range(len(one_style_batch)):
                          source_test_sample_label[i][temp_index_s] = 1
                          target_test_sample_label[i][temp_index_t] = 1
                

                     #get conversion target name ,like pop_piano_test_1
                     target_name = label_enc.inverse_transform([temp_index_t])[0]

                     generated_results,origin_midi,generated_cycle = model.test(sample_images, target_test_sample_label, source_test_sample_label)

                     #print(generated_results.shape)

                     midi_path_origin = os.path.join(file_path, '{}_origin.mid'.format(name))
                     midi_path_transfer = os.path.join(file_path, '{}_transfer_2_{}.mid'.format(name,target_name))
                     midi_path_cycle = os.path.join(file_path, '{}_transfer_2_{}_cycle.mid'.format(name,target_name))

                     save_midis(origin_midi, midi_path_origin)
                     save_midis(generated_results, midi_path_transfer)
                     save_midis(generated_cycle, midi_path_cycle)
                
                
                print('============save converted midis============')

            print('============test finished!============')

        #====================save model=======================

        if epoch % 1 == 0 and epoch != 0:
            print('============save model============')
            model_path = os.path.join(file_path, 'model')

            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)

            print(f'save model: {model_path}')
            model.save(directory=model_path, filename=MODEL_NAME)

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60),
                                                               (time_elapsed_epoch % 60 // 1)))


if __name__ == '__main__':
    processed_dir = './data/rock_bossanova_funk_RnB'
    test_wav_dir = './data/test'


    #parser = argparse.ArgumentParser(description='Train StarGAN music style conversion model.')

    #parser.add_argument('--processed_dir', type=str, help='train dataset directory that contains processed npy and npz files', default=processed_dir)
    #parser.add_argument('--test_wav_dir', type=str, help='test directory that contains raw audios', default=test_wav_dir)

    #argv = parser.parse_args()

    #processed_dir = argv.processed_dir
    #test_wav_dir = argv.test_wav_dir

    start_time = time.time()

    train(processed_dir, test_wav_dir)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Training Time: %02d:%02d:%02d' % \
    (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))



