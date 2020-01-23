import os, shutil
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import numpy as np
import write_midi
from ops import *

processed_dir = './data/rock_bossanova_funk_RnB'




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

    write_midi.write_piano_rolls_to_midi(images_with_pause_list, filename=file_path,

                                         tempo=tempo, beat_resolution=4)




