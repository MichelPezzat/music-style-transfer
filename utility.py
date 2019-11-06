import os, shutil
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import numpy as np
import write_midi
from ops import *

processed_dir = './data/jazz_classic_pop'
#test_wav_dir = './data/test'
#epoch=1


def get_styles(trainset: str = processed_dir,all_styles = []):
    '''return current selected styles for training
        eg. ['pop', 'classic', 'jazz']
    '''
    #p = os.path.join(trainset, "*")
    #all_sub_folder = glob.glob(p)
    

    if '_' in trainset:
    	t = trainset.rsplit('_', maxsplit=1)
    	all_styles.append(t[1])
    	#print(t[0])
    	get_styles(t[0],all_styles)
    else:
    	#print(trainset)
    	all_styles.append(trainset.rsplit('/', maxsplit=1)[1])


    return all_styles

def save_midis(bars, file_path, tempo=80.0):

    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])), bars,

                                  np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))), axis=2)

    pause = np.zeros((bars.shape[0], 64, 128, bars.shape[3]))

    images_with_pause = padded_bars

    images_with_pause = images_with_pause.reshape(-1, 64, padded_bars.shape[2], padded_bars.shape[3])

    images_with_pause_list = []

    for ch_idx in range(padded_bars.shape[3]):

        images_with_pause_list.append(images_with_pause[:, :, :, ch_idx].reshape(images_with_pause.shape[0],

                                                                                 images_with_pause.shape[1],

                                                                                 images_with_pause.shape[2]))

    # write_midi.write_piano_rolls_to_midi(images_with_pause_list, program_nums=[33, 0, 25, 49, 0],

    #                                      is_drum=[False, True, False, False, False], filename=file_path, tempo=80.0)

    write_midi.write_piano_rolls_to_midi(images_with_pause_list, program_nums=[0], is_drum=[False], filename=file_path,

                                         tempo=tempo, beat_resolution=4)
#all_styles = get_styles()
#file_path = os.path.join('./out', f'{epoch}')
#if not os.path.exists(file_path):
#    os.makedirs(file_path)

#tempfiles = []
#for one_style in all_styles:
#    p = os.path.join(test_wav_dir, f'{one_style}/*.npy')
#    npys = glob.glob(p)
#    tempfiles.append(npys[0])
#   tempfiles.append(npys[1])

#for idx in range(len(tempfiles)):
#    sample_npy = np.load(tempfiles[idx]) * 1.
#    sample_npy_re = sample_npy.reshape(1, sample_npy.shape[0], sample_npy.shape[1], 1)
#    midi_path_origin = os.path.join(file_path, '{}_origin.mid'.format(idx + 1))
#    origin_midi = to_binary(sample_npy_re,0.5)
    #print(origin_midi.shape())
#    save_midis(origin_midi, midi_path_origin)





