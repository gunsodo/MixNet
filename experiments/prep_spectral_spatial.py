import argparse
import mixnet.preprocessing as prep
from mixnet.preprocessing.config import CONSTANT

'''

1. In case of preparing a particular dataset, please run the script as an example: python prep_spectral_spatial.py --dataset 'BCIC2a'
2. In case of preparing multiple datasets, please run the script as an example: python prep_spectral_spatial.py --dataset 'BCIC2a' && python prep_spectral_spatial.py --dataset 'BCIC2b' && python prep_spectral_spatial.py --dataset 'BNCI2015_001' && python prep_spectral_spatial.py --dataset 'SMR_BCI' && python prep_spectral_spatial.py --dataset 'HighGamma' && python prep_spectral_spatial.py --dataset 'OpenBMI'

'''
k_folds = 5
pick_smp_freq = 100
n_components = 10 # 10 is the optimal number of CSP components as used in the original paper
bands = [[7.5,14],[11,13],[10,14],[9,12],[19,22],[16,22],[26,34],[17.5,20.5],[7,30],[5,14],[11,31],
         [12,18],[7,9],[15,17],[25,30],[20,25],[5,10],[10,25],[15,30],[10,12],[23,27],[28,32],[12,33],
         [11,22],[5,8],[7.5,17.5],[23,26],[5,20],[5,25],[10,20]]
n_pick_bands = 20 # 20 is the optimal number of selected frequency bands as used in the original paper
order = 5
save_path = 'datasets'
num_class = 2

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='BCIC2a', help='dataset name: ex. [BCIC2a/BCIC2b/BNCI2015_001/SMR_BCI/HighGamma/OpenBMI]')
args = parser.parse_args()

if args.dataset == 'BCIC2a':
    print("============== The {} dataset is being prepared ==========".format(args.dataset))
    prep.BCIC2a.spectral_spatial.subject_dependent_setting(k_folds=k_folds, 
                                                           pick_smp_freq=pick_smp_freq,
                                                           n_components=n_components, 
                                                           bands=bands, 
                                                           n_pick_bands=n_pick_bands, 
                                                           order=order, 
                                                           save_path=save_path, 
                                                           num_class=num_class,
                                                           sel_chs=CONSTANT['BCIC2a']['sel_chs'])
    
    prep.BCIC2a.spectral_spatial.subject_independent_setting(k_folds=k_folds, 
                                                             pick_smp_freq=pick_smp_freq,
                                                             n_components=n_components, 
                                                             bands=bands, 
                                                             n_pick_bands=n_pick_bands, 
                                                             order=order, 
                                                             save_path=save_path, 
                                                             num_class=num_class,
                                                             sel_chs=CONSTANT['BCIC2a']['sel_chs'])
    
elif args.dataset == 'BCIC2b':
    print("============== The {} dataset is being prepared ==========".format(args.dataset))
    prep.BCIC2b.spectral_spatial.subject_dependent_setting(k_folds=k_folds, 
                                                           pick_smp_freq=pick_smp_freq,
                                                           n_components=n_components, 
                                                           bands=bands, 
                                                           n_pick_bands=n_pick_bands, 
                                                           order=order, 
                                                           save_path=save_path, 
                                                           num_class=num_class,
                                                           sel_chs=CONSTANT['BCIC2b']['sel_chs'])
    
    prep.BCIC2b.spectral_spatial.subject_independent_setting(k_folds=k_folds, 
                                                             pick_smp_freq=pick_smp_freq,
                                                             n_components=n_components, 
                                                             bands=bands, 
                                                             n_pick_bands=n_pick_bands, 
                                                             order=order, 
                                                             save_path=save_path, 
                                                             num_class=num_class,
                                                             sel_chs=CONSTANT['BCIC2b']['sel_chs'])

elif args.dataset == 'BNCI2015_001':
    print("============== The {} dataset is being prepared ==========".format(args.dataset))
    prep.BNCI2015_001.spectral_spatial.subject_dependent_setting(k_folds=k_folds, 
                                                            pick_smp_freq=pick_smp_freq,
                                                            n_components=n_components, 
                                                            bands=bands, 
                                                            n_pick_bands=n_pick_bands, 
                                                            order=order, 
                                                            save_path=save_path, 
                                                            num_class=num_class,
                                                            sel_chs=CONSTANT['BNCI2015_001']['sel_chs'])
    
    prep.BNCI2015_001.spectral_spatial.subject_independent_setting(k_folds=k_folds, 
                                                              pick_smp_freq=pick_smp_freq,
                                                              n_components=n_components, 
                                                              bands=bands, 
                                                              n_pick_bands=n_pick_bands, 
                                                              order=order, 
                                                              save_path=save_path, 
                                                              num_class=num_class,
                                                              sel_chs=CONSTANT['BNCI2015_001']['sel_chs'])

if args.dataset == 'SMR_BCI':  
    print("============== The {} dataset is being prepared ==========".format(args.dataset))
    prep.SMR_BCI.spectral_spatial.subject_dependent_setting(k_folds=k_folds, 
                                                            pick_smp_freq=pick_smp_freq,
                                                            n_components=n_components, 
                                                            bands=bands, 
                                                            n_pick_bands=n_pick_bands, 
                                                            order=order, 
                                                            save_path=save_path, 
                                                            num_class=num_class,
                                                            sel_chs=CONSTANT['SMR_BCI']['sel_chs'])
    
    prep.SMR_BCI.spectral_spatial.subject_independent_setting(k_folds=k_folds, 
                                                              pick_smp_freq=pick_smp_freq,
                                                              n_components=n_components, 
                                                              bands=bands, 
                                                              n_pick_bands=n_pick_bands, 
                                                              order=order, 
                                                              save_path=save_path, 
                                                              num_class=num_class,
                                                              sel_chs=CONSTANT['SMR_BCI']['sel_chs'])
    
elif args.dataset == 'HighGamma':
    print("============== The {} dataset is being prepared ==========".format(args.dataset))    
    prep.HighGamma.spectral_spatial.subject_dependent_setting(k_folds=k_folds, 
                                                            pick_smp_freq=pick_smp_freq,
                                                            n_components=n_components, 
                                                            bands=bands, 
                                                            n_pick_bands=n_pick_bands, 
                                                            order=order, 
                                                            save_path=save_path, 
                                                            num_class=num_class,
                                                            sel_chs=CONSTANT['HighGamma']['sel_chs'])
    
    prep.HighGamma.spectral_spatial.subject_independent_setting(k_folds=k_folds, 
                                                              pick_smp_freq=pick_smp_freq,
                                                              n_components=n_components, 
                                                              bands=bands, 
                                                              n_pick_bands=n_pick_bands, 
                                                              order=order, 
                                                              save_path=save_path, 
                                                              num_class=num_class,
                                                              sel_chs=CONSTANT['HighGamma']['sel_chs'])

elif args.dataset == 'OpenBMI': 
    print("============== The {} dataset is being prepared ==========".format(args.dataset))
    prep.OpenBMI.spectral_spatial.subject_dependent_setting(k_folds=k_folds, 
                                                            pick_smp_freq=pick_smp_freq,
                                                            n_components=n_components, 
                                                            bands=bands, 
                                                            n_pick_bands=n_pick_bands, 
                                                            order=order, 
                                                            save_path=save_path, 
                                                            num_class=num_class,
                                                            sel_chs=CONSTANT['OpenBMI']['sel_chs'])
    
    prep.OpenBMI.spectral_spatial.subject_independent_setting(k_folds=k_folds, 
                                                              pick_smp_freq=pick_smp_freq,
                                                              n_components=n_components, 
                                                              bands=bands, 
                                                              n_pick_bands=n_pick_bands, 
                                                              order=order, 
                                                              save_path=save_path, 
                                                              num_class=num_class,
                                                              sel_chs=CONSTANT['OpenBMI']['sel_chs'])
    
else:
    raise Exception('Path Error: {} does not exist, please correct the dataset name.'.format(args.dataset))