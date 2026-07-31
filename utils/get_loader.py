from utils.bcichaDataLoader import BCIchaDataLoader, BCIchaDataLoader_fixed
from utils.bcichaDataLoaderFull import BCIchaDataLoaderFull, BCIchaDataLoaderFull_fixed
from utils.bciDataLoaderFull import BCIDataLoaderFull
from utils.bciDataLoader import BCIDataLoader

from utils.mamemDataloader import MAMEMDataLoader
from utils.mamemDataloaderFull import MAMEMDataLoaderFull


def get_datasets_args(args):
    return get_datasets(args['dataset'],
                        args['sub'],
                        args['finetune'],
                        args['bs'],
                        args['subject_weights'],
                        args['model'])


def get_datasets(dataset_name, sub, finetune, bs, subject_weights, model_name):
    if dataset_name == 'bci':
        in_channels = 22
        num_pred_classes = 4
        num_subjects = 9
        data_path = './data/BCICIV_2a_mat/'
        if sub == 'all' and not finetune:
            dataloader = BCIDataLoaderFull(subject=sub, ratio=8, data_path=data_path, bs=bs,
                                           model=model_name, subject_weights=subject_weights)
        else:
            dataloader = BCIDataLoader(subject=sub, ratio=8, data_path=data_path, bs=bs,
                                       model=model_name, finetune=finetune)
        trainloader, validloader, testloader = dataloader.get_dataloader()
    
    elif dataset_name.startswith('mamem'):
        in_channels = 8
        num_pred_classes = 5
        num_subjects = 11
        data_path = './data/MAMEM/'
    
        if sub == 'all' and not finetune:
            dataloader = MAMEMDataLoaderFull(subject=sub, ratio=8, data_path=data_path,
                                             bs=bs,
                                             model=model_name, subject_weights=subject_weights)
        else:
            dataloader = MAMEMDataLoader(subject=sub, ratio=8, data_path=data_path, bs=bs,
                                         model=model_name, finetune=finetune)
        trainloader, validloader, testloader = dataloader.get_dataloader()
    
    elif dataset_name.startswith('bcicha'):
        in_channels = 56
        num_pred_classes = 2
        num_subjects = 16
        data_path = './data/BCIcha/'
        if dataset_name.endswith('fixed'):
            if sub == 'all 'and not finetune:
                dataloader = BCIchaDataLoaderFull_fixed(subject=sub, data_path=data_path, bs=bs,
                                                        model=model_name, subject_weights=subject_weights)
            else:
                dataloader = BCIchaDataLoader_fixed(subject=sub, data_path=data_path, bs=bs,
                                                    model=model_name)
            trainloader, validloader, testloader = dataloader.get_dataloader()
        else:
            if sub == 'all' and not finetune:
                dataloader = BCIchaDataLoaderFull(subject=sub, data_path=data_path, bs=bs,
                                                  model=model_name, subject_weights=subject_weights)
            else:
                dataloader = BCIchaDataLoader(subject=sub, data_path=data_path, bs=bs,
                                              model=model_name, finetune=finetune)
            trainloader, validloader, testloader = dataloader.get_dataloader()
    else:
        raise ValueError('No such dataset')

    return trainloader, validloader, testloader, in_channels, num_pred_classes, num_subjects
