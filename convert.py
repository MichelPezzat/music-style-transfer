import argparse
import os
import numpy as np

from model import StarGAN
from preprocess import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utility import *

#get all speaker

all_styles = get_styles(trainset= './data/rock_bossanova_funk_RnB',all_styles = [])
label_enc = LabelEncoder()
label_enc.fit(all_styles)


def conversion(model_dir, test_dir, output_dir, source, target):
    if not os.path.exists(model_dir) or not os.path.exists(test_dir):
        raise Exception('model dir or test dir not exist!')
    


    model = StarGAN(time_step=TIME_STEP, pitch_range=PITCH_RANGE,styles_num =STYLES_NUM,batchsize = BATCHSIZE)
    model.load(filepath)

    #f'./data/fourspeakers_test/{source}/*.wav'
    p = os.path.join(test_dir, f'{source}/*.npy')
    phrases = glob.glob(p)
    phrases.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

    

    for one_style_phrase in phrases:
        _, style, name = one_style_phrase[0].rsplit('/', maxsplit=2)

        sample_images = np.load(one_style_phrase[0])*1 
        sample_images = np.array(sample_images).astype(np.float32)

            
        one_test_sample_label = np.zeros([len(one_style_phrase),len(all_styles)])
        temp_index = label_enc.transform([target])[0]
        one_test_sample_label[:,temp_index]=1

                
        
                

                #get conversion target name ,like pop_piano_test_1
        target_name = label_enc.inverse_transform([temp_index])[0]

        generated_results,origin_midi = model.test(sample_images, one_test_sample_label)

        #print(generated_results.shape)

        midi_path_origin = os.path.join(file_path, '{}_origin.mid'.format(name))
        midi_path_transfer = os.path.join(file_path, '{}_transfer_2_{}.mid'.format(name,target_name))

        npy_path_test = os.path.join(file_path, '{}_transfer_2_{}.npy'.format(name,target_name))

        save_midis(origin_midi, midi_path_origin)
        save_midis(generated_results, midi_path_transfer)

        npy_path_origin = os.path.join(npy_path_test, 'origin')
        npy_path_transfer = os.path.join(npy_path_test, 'transfer')

        if not os.path.exists(npy_path_origin):
                os.makedirs(npy_path_origin)
        if not os.path.exists(npy_path_transfer):
            os.makedirs(npy_path_transfer)

        np.save(os.path.join(npy_path_origin, '{}_origin.npy'.format(name)), origin)
        np.save(os.path.join(npy_path_transfer, '{}_transfer.npy'.format(name,target_name)), generated_results)





                

                
                
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert voices using pre-trained CycleGAN model.')

    model_dir = './out/90_2018-10-17-22-58/model/'
    test_dir = './data/test/'
    source_genre = 'rock'
    target_genre = 'RnB'
    output_dir = './converted_midis'

    parser.add_argument('--model_dir', type=str, help='Directory for the pre-trained model.', default=model_dir)
    parser.add_argument('--test_dir', type=str, help='Directory for the voices for conversion.', default=test_dir)
    parser.add_argument('--output_dir', type=str, help='Directory for the converted voices.', default=output_dir)
    parser.add_argument('--source_genre', type=str, help='source_speaker', default=source_speaker)
    parser.add_argument('--target_genre', type=str, help='target_speaker', default=target_speaker)

    argv = parser.parse_args()

    model_dir = argv.model_dir
    test_dir = argv.test_dir
    output_dir = argv.output_dir
    source_speaker = argv.source_speaker
    target_speaker = argv.target_speaker

    conversion(model_dir = model_dir,\
     test_dir = test_dir, output_dir = output_dir, source=source_genre, target=target_genre)