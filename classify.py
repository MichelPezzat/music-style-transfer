import argparse
import numpy as np
import tensorflow as tf
from random import shuffle
#from collections import namedtuple
from module import *
from ops import *
#from utils import *
from utility import *


from glob import glob

all_styles = get_styles(trainset= './data/rock_bossanova_funk_RnB',all_styles = [])
label_enc = LabelEncoder()
label_enc.fit(all_styles)

def classification(model_dir, test_dir, output_dir, source, target):

        
        # load the origin samples in npy format and sorted in ascending order
        sample_files_origin = glob('{}/{}2{}npys/origin/*.*'.format(test_dir,source,target))
        sample_files_origin.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[0]))

        # load the origin samples in npy format and sorted in ascending order
        sample_files_transfer = glob('{}/{}2{}npys/transfer/*.*'.format(test_dir,source,target))
        sample_files_transfer.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[0]))

        # load the origin samples in npy format and sorted in ascending order
        sample_files_cycle = glob('{}/{}2{}npys/cycle/*.*'.format(test_dir,source,target))
        sample_files_cycle.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[0]))

        # put the origin, transfer and cycle of the same phrase in one zip
        sample_files = list(zip(sample_files_origin, sample_files_transfer, sample_files_cycle))

        model = StarGAN(time_step=TIME_STEP, mode = 'test')
        model.load(filepath=model_dir)

        # create a test path to store the generated sample midi files attached with probability
        #test_dir_mid = os.path.join(args.test_dir, '{}2{}_{}_{}_{}/{}/mid_attach_prob'.format(self.dataset_A_dir,
         #                                                                                     self.dataset_B_dir,
          #                                                                                    self.model,
           #                                                                                   self.sigma_d,
            #                                                                                  self.now_datetime,
             #                                                                                 args.which_direction))
        #if not os.path.exists(test_dir_mid):
         #   os.makedirs(test_dir_mid)

        count_origin = 0
        count_transfer = 0
        count_cycle = 0
        line_list = []

        one_test_sample_label_s = np.zeros([1,len(all_styles)])

        temp_index_s = label_enc.transform([source])[0]
        one_test_sample_label_s[:,temp_index]= 1

        one_test_sample_label_t = np.zeros([1,len(all_styles)])

        temp_index_t = label_enc.transform([target])[0]
        one_test_sample_label_t[:,temp_index]= 1



        for idx in range(len(sample_files)):
            print('Classifying midi: ', sample_files[idx])

            # load sample phrases in npy formats
            sample_origin = np.load(sample_files[idx][0])
            sample_transfer = np.load(sample_files[idx][1])
            sample_cycle = np.load(sample_files[idx][2])

            # get the probability for each sample phrase
            test_result_origin = model.test_classifier(sample_origin)
            test_result_transfer = model.test_classifier(sample_transfer)
            test_result_cycle = model.test_classifier(sample_transfer)

            origin_transfer_diff = np.abs(test_result_origin - test_result_transfer)
            content_diff = np.mean((sample_origin * 1.0 - sample_transfer * 1.0)**2)



            # labels: (1, 0) for A, (0, 1) for B
            for style_index in zip(temp_index_s,temp_index_t):
                line_list.append((idx + 1, content_diff, origin_transfer_diff[0][style_index], test_result_origin[0][style_index],
                                  test_result_transfer[0][style_index], test_result_cycle[0][style_index]))

                # for the accuracy calculation
                count_origin += 1 if np.argmax(test_result_origin[0]) == 0 else 0
                count_transfer += 1 if np.argmax(test_result_transfer[0]) == 0 else 0
                count_cycle += 1 if np.argmax(test_result_cycle[0]) == 0 else 0

                # create paths for origin, transfer and cycle samples attached with probability
                

            

                
               

            
            

        # sort the line_list based on origin_transfer_diff and write to a ranking txt file
        line_list.sort(key=lambda x: x[2], reverse=True)
        with open(os.path.join(output_dir, 'Rankings_{}2{}.txt'.format(source,target)), 'w') as f:
            f.write('Id  Content_diff  P_O - P_T  Prob_Origin  Prob_Transfer  Prob_Cycle')
            for i in range(len(line_list)):
                f.writelines("\n%5d %5f %5f %5f %5f %5f" % (line_list[i][0], line_list[i][1], line_list[i][2],
                                                            line_list[i][3], line_list[i][4], line_list[i][5]))
        f.close()

        # calculate the accuracy
        accuracy_origin = count_origin * 1.0 / len(sample_files)
        accuracy_transfer = count_transfer * 1.0 / len(sample_files)
        accuracy_cycle = count_cycle * 1.0 / len(sample_files)
        print('Accuracy of this classifier on test datasets is :', accuracy_origin, accuracy_transfer, accuracy_cycle)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert voices using pre-trained CycleGAN model.')

    model_dir = './out/100_2020-01-24-12-30/model'
    test_dir = './converted_midis_and_npys'
    source_genre = 'rock'
    target_genre = 'RnB'
    output_dir = './results'

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

    classification(model_dir = model_dir,\
     test_dir = test_dir, output_dir = output_dir, source=source_genre, target=target_genre)