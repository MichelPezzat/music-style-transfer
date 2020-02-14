import argparse
import os
import numpy as np

from model import StarGAN
#from preprocess import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utility import *

#get all speaker



all_styles = get_styles(trainset= './data/rock_bossanova_funk_RnB',all_styles = [])
label_enc = LabelEncoder()
label_enc.fit(all_styles)


def conversion(model_dir, test_dir, output_dir, source, target):
    if not os.path.exists(model_dir) or not os.path.exists(test_dir):
        raise Exception('model dir or test dir not exist!')


    


    model = StarGAN(time_step=TIME_STEP, pitch_range =PITCH_RANGE,styles_num = STYLES_NUM, mode = 'test')
    model.load(filepath=model_dir)


    #f'./data/fourspeakers_test/{source}/*.wav'
    p = os.path.join(test_dir, f'{source}/*.npy')
    phrases = glob.glob(p)
    phrases.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

    #print(len(phrases))

    

    for one_style_phrase in phrases:
        
        _, style, name = one_style_phrase.rsplit('\\', maxsplit=2)

        

        sample_images = np.load(one_style_phrase)*1 
        sample_images = np.array(sample_images).astype(np.float32)
        sample_images_re = sample_images.reshape(1, sample_images.shape[0], sample_images.shape[1], sample_images.shape[2])

            
        source_test_sample_label = np.zeros([1,len(all_styles)])
        target_test_sample_label = np.zeros([1,len(all_styles)])
        temp_index_s = label_enc.transform([target])[0]
        temp_index_t = label_enc.transform([target])[0]
        source_test_sample_label[:,temp_index_s]= 1

        target_test_sample_label[:,temp_index_t]= 1

                
        
                
                
        target_name = label_enc.inverse_transform([temp_index_t])[0]

        generated_results, origin_midi, generated_cycle = model.test(sample_images_re, target_test_sample_label, source_test_sample_label)

        midipath = f'{output_dir}/{style}2{target_name}_midis'
        npypath = f'{output_dir}/{style}2{target_name}npys'


        if not os.path.exists(midipath):
            os.makedirs(midipath, exist_ok=True)
        if not os.path.exists(npypath):
            os.makedirs(npypath, exist_ok=True)



        midi_path_origin = os.path.join(midipath, '{}_origin.mid'.format(name))
        midi_path_transfer = os.path.join(midipath, '{}_transfer_2_{}.mid'.format(name,target_name))
        midi_path_cycle = os.path.join(midipath, '{}_cycle_2_{}.mid'.format(name,target_name))


        save_midis(origin_midi, midi_path_origin)
        save_midis(generated_results, midi_path_transfer)
        save_midis(generated_cycle, midi_path_cycle)

        npy_path_origin = os.path.join(npypath, 'origin')
        npy_path_transfer = os.path.join(npypath, 'transfer')
        npy_path_cycle = os.path.join(npypath, 'cycle')


        if not os.path.exists(npy_path_origin):
                os.makedirs(npy_path_origin)
        if not os.path.exists(npy_path_transfer):
            os.makedirs(npy_path_transfer)
        if not os.path.exists(npy_path_cycle):
            os.makedirs(npy_path_cycle)

        np.save(os.path.join(npy_path_origin, '{}_origin.npy'.format(name)), origin_midi)
        np.save(os.path.join(npy_path_transfer, '{}_transfer_2_{}.npy'.format(name,target_name)), generated_results)
        np.save(os.path.join(npy_path_cycle, '{}_transfer_2_{}_cycle.npy'.format(name,target_name)), generated_cycle)





                

                
                
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert voices using pre-trained CycleGAN model.')

    model_dir = './out/100_2020-01-24-12-30/model'
    test_dir = './data/test'
    source_genre = 'rock'
    target_genre = 'RnB'
    output_dir = './converted_midis_and_npys'

    parser.add_argument('--model_dir', type=str, help='Directory for the pre-trained model.', default=model_dir)
    parser.add_argument('--test_dir', type=str, help='Directory for the voices for conversion.', default=test_dir)
    parser.add_argument('--output_dir', type=str, help='Directory for the converted voices.', default=output_dir)
    parser.add_argument('--source_genre', type=str, help='source_speaker', default=source_genre)
    parser.add_argument('--target_genre', type=str, help='target_speaker', default=target_genre)

    argv = parser.parse_args()

    model_dir = argv.model_dir
    test_dir = argv.test_dir
    output_dir = argv.output_dir
    source_genre = argv.source_genre
    target_genre = argv.target_genre

    conversion(model_dir = model_dir,\
     test_dir = test_dir, output_dir = output_dir, source=source_genre, target=target_genre)