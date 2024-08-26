import argparse
import mixnet.preprocessing as prep
from mixnet.preprocessing.config import CONSTANT

'''

1. In case of preparing a particular dataset, please run the script as an example: python prep_spectral_spatial_signals.py --dataset 'BCIC2a'
2. In case of preparing multiple datasets, please run the script as an example: python prep_spectral_spatial_signals.py --dataset 'BCIC2a' && python prep_spectral_spatial_signals.py --dataset 'BCIC2b' && python prep_spectral_spatial_signals.py --dataset 'BNCI2015_001' && python prep_spectral_spatial_signals.py --dataset 'SMR_BCI' && python prep_spectral_spatial_signals.py --dataset 'HighGamma' && python prep_spectral_spatial_signals.py --dataset 'OpenBMI'

'''
k_folds = 5
pick_smp_freq = 100
bands = [[4, 8], [8, 12], [12, 16], 
            [16, 20], [20, 24], [24, 28], 
            [28, 32], [32, 36], [36, 40]]
order = 5
save_path = 'datasets'
num_class = 2

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='BCIC2a', help='dataset name: ex. [BCIC2a/BCIC2b/BNCI2015_001/SMR_BCI/HighGamma/OpenBMI]')
args = parser.parse_args()

if args.dataset == 'BCIC2a':
    print("============== The {} dataset is being prepared ==========".format(args.dataset))
    prep.BCIC2a.spectral_spatial_signals.subject_dependent_setting(k_folds=k_folds,
                                                  pick_smp_freq=pick_smp_freq, 
                                                  n_components=2, # 2 is the optimal number of CSP components 
                                                  bands=bands, 
                                                  order=order, 
                                                  save_path=save_path, 
                                                  num_class=num_class, 
                                                  sel_chs=CONSTANT['BCIC2a']['sel_chs'])
    
    prep.BCIC2a.spectral_spatial_signals.subject_independent_setting(k_folds=k_folds,
                                                pick_smp_freq=pick_smp_freq, 
                                                n_components=4, # 4 is the optimal number of CSP components
                                                bands=bands, 
                                                order=order, 
                                                save_path=save_path, 
                                                num_class=num_class, 
                                                sel_chs=CONSTANT['BCIC2a']['sel_chs'])
    
elif args.dataset == 'BCIC2b':  
    print("============== The {} dataset is being prepared ==========".format(args.dataset))
    prep.BCIC2b.spectral_spatial_signals.subject_dependent_setting(k_folds=k_folds,
                                                  pick_smp_freq=pick_smp_freq, 
                                                  n_components=2, # 2 is the optimal number of CSP components 
                                                  bands=bands, 
                                                  order=order, 
                                                  save_path=save_path, 
                                                  num_class=num_class, 
                                                  sel_chs=CONSTANT['BCIC2b']['sel_chs'])
    
    prep.BCIC2b.spectral_spatial_signals.subject_independent_setting(k_folds=k_folds,
                                                pick_smp_freq=pick_smp_freq, 
                                                n_components=2, # 2 is the optimal number of CSP components 
                                                bands=bands, 
                                                order=order, 
                                                save_path=save_path, 
                                                num_class=num_class, 
                                                sel_chs=CONSTANT['BCIC2b']['sel_chs'])
    
elif args.dataset == 'BNCI2015_001':
    print("============== The {} dataset is being prepared ==========".format(args.dataset))
    prep.BNCI2015_001.spectral_spatial_signals.subject_dependent_setting(k_folds=k_folds,
                                                   pick_smp_freq=pick_smp_freq, 
                                                   n_components=4, # 4 is the optimal number of CSP components
                                                   bands=bands, 
                                                   order=order, 
                                                   save_path=save_path, 
                                                   num_class=num_class, 
                                                   sel_chs=CONSTANT['BNCI2015_001']['sel_chs'])
    
    prep.BNCI2015_001.spectral_spatial_signals.subject_independent_setting(k_folds=k_folds,
                                                 pick_smp_freq=pick_smp_freq, 
                                                 n_components=2, # 2 is the optimal number of CSP components
                                                 bands=bands, 
                                                 order=order, 
                                                 save_path=save_path, 
                                                 num_class=num_class, 
                                                 sel_chs=CONSTANT['BNCI2015_001']['sel_chs'])
    
elif args.dataset == 'SMR_BCI':  
    print("============== The {} dataset is being prepared ==========".format(args.dataset))
    prep.SMR_BCI.spectral_spatial_signals.subject_dependent_setting(k_folds=k_folds,
                                             pick_smp_freq=pick_smp_freq, 
                                             n_components=2, # 2 is the optimal number of CSP components
                                             bands=bands, 
                                             order=order, 
                                             save_path=save_path, 
                                             num_class=num_class, 
                                             sel_chs=CONSTANT['SMR_BCI']['sel_chs'])

    prep.SMR_BCI.spectral_spatial_signals.subject_independent_setting(k_folds=k_folds,
                                               pick_smp_freq=pick_smp_freq, 
                                               n_components=6, # 6 is the optimal number of CSP components
                                               bands=bands, 
                                               order=order, 
                                               save_path=save_path, 
                                               num_class=num_class, 
                                               sel_chs=CONSTANT['SMR_BCI']['sel_chs'])
    
elif args.dataset == 'HighGamma':
    print("============== The {} dataset is being prepared ==========".format(args.dataset))
    prep.HighGamma.spectral_spatial_signals.subject_dependent_setting(k_folds=k_folds,
                                                   pick_smp_freq=pick_smp_freq, 
                                                   n_components=6, # 6 is the optimal number of CSP components
                                                   bands=bands, 
                                                   order=order, 
                                                   save_path=save_path, 
                                                   num_class=num_class, 
                                                   sel_chs=CONSTANT['HighGamma']['sel_chs'])

    prep.HighGamma.spectral_spatial_signals.subject_independent_setting(k_folds=k_folds,
                                                 pick_smp_freq=pick_smp_freq, 
                                                 n_components=10, # 10 is the optimal number of CSP components
                                                 bands=bands, 
                                                 order=order, 
                                                 save_path=save_path, 
                                                 num_class=num_class, 
                                                 sel_chs=CONSTANT['HighGamma']['sel_chs'])
    
elif args.dataset == 'OpenBMI': 
    print("============== The {} dataset is being prepared ==========".format(args.dataset))
    prep.OpenBMI.spectral_spatial_signals.subject_dependent_setting(k_folds=k_folds,
                                                 pick_smp_freq=pick_smp_freq, 
                                                 n_components=4, # 4 is the optimal number of CSP components
                                                 bands=bands, 
                                                 order=order, 
                                                 save_path=save_path, 
                                                 num_class=num_class, 
                                                 sel_chs=CONSTANT['OpenBMI']['sel_chs'])

    prep.OpenBMI.spectral_spatial_signals.subject_independent_setting(k_folds=k_folds,
                                                   pick_smp_freq=pick_smp_freq, 
                                                   n_components=2, # 2 is the optimal number of CSP components
                                                   bands=bands, 
                                                   n_features=n_features,
                                                   order=order, 
                                                   save_path=save_path, 
                                                   num_class=num_class, 
                                                   sel_chs=CONSTANT['OpenBMI']['sel_chs'])
    
else:
    raise Exception('Path Error: {} does not exist, please correct the dataset name.'.format(args.dataset))
