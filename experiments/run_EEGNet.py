import tensorflow as tf
import numpy as np
import os
import argparse
from mixnet.data_loader import DataLoader
from mixnet.utils import write_log
from configs import exp_config
'''

In case of evaluating classification performance on a particular dataset, please run the script as an example: 

python run_EEGNet.py --model_name 'EEGNet' --dataset 'BCIC2a' --train_type 'subject_dependent' --data_type 'time_domain' --num_class 2  --num_chs 20 --GPU 0


'''
def main(subject):
    # create an object of DataLoader
    loader = DataLoader(subject=subject, **config.data_params)

    results = []
    for fold in range(1, config.data_params.n_folds+1):

        prefix_log = 'S{:03d}_fold{:02d}'.format(subject, fold)
        model = config.model(prefix_log=prefix_log, **config.model_params)
        
        # load dataset
        X_train, y_train = loader.load_train_set(fold=fold)
        X_val, y_val = loader.load_val_set(fold=fold)
        X_test, y_test = loader.load_test_set(fold=fold)
        model.fit(X_train, y_train, X_val, y_val)
        Y, evaluation = model.evaluate(X_test, y_test)

        # logging
        csv_file = config.log_path+'/S{:03d}_all_results.csv'.format(subject)
        if fold==1:
            write_log(csv_file, data=evaluation.keys(), mode='w')
        write_log(csv_file, data=evaluation.values(), mode='a')
        results.append(Y)
        tf.keras.backend.clear_session()
        
    # writing results
    np.save(config.log_path+'/S{:03d}_prediction_results.npy'.format(subject), results)
    print('------------------------- S{:03d} Done--------------------------'.format(subject))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='EEGNet', help='model name')
    parser.add_argument('--dataset', type=str, default='BCIC2a', help='dataset name: ex. [BCIC2a/BCIC2b/BNCI2015_001/SMR_BCI/HighGamma/OpenBMI]')
    parser.add_argument('--train_type', type=str, default='subject_dependent', help='Train type: ex. subject_dependent, subject_independent')
    parser.add_argument('--data_type', type=str, default='time_domain', help='Train type: ex. time_domain')
    parser.add_argument('--num_class', type=int, default=2, help='number of classes')
    parser.add_argument('--num_chs', type=int, default=20, help='number of classes')
    parser.add_argument('--loss_weights', nargs='+', default=None, type=float, help='loss_weights (beta): ex. [beta1,beta2,beta3,...]')
    parser.add_argument('--log_dir', type=str, default='logs', help='path to save logs')
    parser.add_argument('--subjects', nargs='+', default=None, type=int, help='list of range test subject, None=all subject')
    parser.add_argument('--GPU', type=str, default='0', help='GPU ID')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
    print(tf.config.experimental.list_physical_devices('GPU'))

    exp_setup = {'dataset': args.dataset, 
                 'train_type': args.train_type, 
                 'data_type': args.data_type,
                 'num_class': args.num_class,
                 'num_chs': args.num_chs,
                 'loss_weights': args.loss_weights,
                 'log_dir': args.log_dir}
    config = exp_config.get_params(args.model_name, **exp_setup)
    print(config)                                
    for directory in [config.log_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    if args.subjects == None: #loop to train all subjects
        for subject in range(1, config.data_params.n_subjects+1):
            main(subject)
    elif len(args.subjects)==2:
        for subject in range(args.subjects[0], args.subjects[1]+1):
            main(subject)
    else:
        for subject in args.subjects:
            main(subject)
