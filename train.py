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
STYLES_NUM = 3

def get_files_labels(pattern: str):
    files = glob.glob(pattern)
    names = []
    for f in files:
        t = f.rsplit('/', maxsplit=1)[1]  #'./data/jazz_classic_pop/jazz_piano_train_273.npy'
        name = t.rsplit('.', maxsplit=1)[0]
        names.append(name)

    return files, names


def train(processed_dir: str, test_wav_dir: str):
    timestr = time.strftime("%Y-%m-%d-%H-%M", time.localtime())  #like '2018-10-10-14-47'

    all_styles = get_styles(processed_dir)
    label_enc = LabelEncoder()
    label_enc.fit(all_styles)

    lambda_cycle = 10
    lambda_identity = 5
    lambda_classifier = 3

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
    BATCHSIZE = 16
    model = StarGANVC(time_step=TIME_STEP, pitch_range=PITCH_RANGE,styles_num =STYLES_NUM,batchsize = BATCHSIZE)
    #====================start train==============#
    EPOCH = 101

    num_samples = len(files)
    for epoch in range(EPOCH):
        start_time_epoch = time.time()

        files_shuffled, names_shuffled = shuffle(files, names)

        for i in range(num_samples // BATCHSIZE):
            num_iterations = num_samples // BATCHSIZE * epoch + i

            if num_iterations > 2500:
                
                domain_classifier_learning_rate = max(0, domain_classifier_learning_rate - domain_classifier_learning_rate_decay)
                generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
                discriminator_learning_rate = max(0, discriminator_learning_rate - discriminator_learning_rate_decay)

            if discriminator_learning_rate == 0 or generator_learning_rate == 0:
                print('Early stop training.')
                break
            # if num_iterations > 2500:
            #     lambda_identity = 1
            #     domain_classifier_learning_rate = 0
            #     generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
            #     discriminator_learning_rate = discriminator_learning_rate + discriminator_learning_rate_decay

            # if generator_learning_rate <= 0.0001:
            #     generator_learning_rate = 0.0001
            # if discriminator_learning_rate >= 0.0002:
            #     discriminator_learning_rate = 0.0002

            start = i * BATCHSIZE
            end = (i + 1) * BATCHSIZE

            if end > num_samples:
                end = num_samples

            X, X_t, y, y_t = [], [], [], []

            #get target file paths
            batchnames = names_shuffled[start:end]
            pre_targets = []
            for name in batchnames:
                name = name.split(sep='_')[0]  #pop
                t = np.random.choice(exclude_dict[name], 1)[0]
                pre_targets.append(t)
            #one batch train data
            for one_filename, one_name, one_target in zip(files_shuffled[start:end], names_shuffled[start:end], pre_targets):

                #target name
                t = one_target.rsplit('/', maxsplit=1)[1]  #'./data/jazz_classic_pop/pop_piano_train_688.npy'
                target_style_name = t.rsplit('.', maxsplit=1)[0].split('_')[0]

                #source name
                style_name = one_name.split('_')[0]  #pop

                #shape [1,64,84,1]
                one_file = np.load(one_filename)
                
                X.append(one_file)

                #source label
                temp_index = label_enc.transform([style_name])[0]
                temp_arr_s = np.zeros([
                    len(all_styles),
                ])
                temp_arr_s[temp_index] = 1
                y.append(temp_arr_s)

                #load target files and labels
                one_file_t = np.load(one_target).astype(np.float32)
                

                #[1,84,64,1]
                
                X_t.append(one_file_t)

                #target label
                temp_index_t = label_enc.transform([target_style_name])[0]
                temp_arr_t = np.zeros([
                    len(all_styles),
                ])
                temp_arr_t[temp_index_t] = 1
                y_t.append(temp_arr_t)


            generator_loss, discriminator_loss, domain_classifier_loss = model.train(\
            input_source=X, input_target=X_t, source_label=y, \
            target_label=y_t, generator_learning_rate=generator_learning_rate,\
             discriminator_learning_rate=discriminator_learning_rate,\
            classifier_learning_rate=domain_classifier_learning_rate, \
            lambda_identity=lambda_identity, lambda_cycle=lambda_cycle,\
            lambda_classifier=lambda_classifier
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
                tempfiles.append(list(zip(npys[:BATCHSIZE])))   #'./data/test/pop/pop_piano_test.npy'

            for one_style_batch in tempfiles:
                _, style, name = one_style_batch[0][0].rsplit('/', maxsplit=2)

                sample_images = [np.load(one_style_batch[0])*1 for one_style_batch in one_style_batch]
                sample_images = np.array(sample_images).astype(np.float32)

                #target label 1->2, 2->3, 3->0, 0->1
                one_test_sample_label = np.zeros([len(all_styles)])
                temp_index = label_enc.transform([style_name])[0]
                temp_index = (temp_index + 2) % len(all_styles)

                
                one_test_sample_label[temp_index] = 1

                #get conversion target name ,like SF1
                target_name = label_enc.inverse_transform([temp_index])[0]

                generated_results,origin_midi = model.test(sample_images, one_test_sample_label)

                midi_path_origin = os.path.join(file_path, '{}_origin.mid'.format(name))
                midi_path_transfer = os.path.join(file_path, '{}_transfer_2_{}.mid'.format(name,target_name))

                save_midis(origin_midi, midi_path_origin)
                save_midis(generated_results_binary, midi_path_transfer)
                
                
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
    processed_dir = './data/jazz_classic_pop'
    test_wav_dir = './data/test'


    parser = argparse.ArgumentParser(description='Train StarGAN music style conversion model.')

    parser.add_argument('--processed_dir', type=str, help='train dataset directory that contains processed npy and npz files', default=processed_dir)
    parser.add_argument('--test_wav_dir', type=str, help='test directory that contains raw audios', default=test_wav_dir)

    argv = parser.parse_args()

    processed_dir = argv.processed_dir
    test_wav_dir = argv.test_wav_dir

    start_time = time.time()

    train(processed_dir, test_wav_dir)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Training Time: %02d:%02d:%02d' % \
    (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

