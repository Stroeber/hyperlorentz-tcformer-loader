https://pytorchtime.com/docs/stable/tutorials/classification_tutorial.html Inception time 

Could you please generate a python hyperparameter (HP) search script, do hyperparameter search for my time series classification model. We can call "main.py" to train the model with the given parameters. I have a list of default parameters for each of the three Datasets I'm using. However, for the HP Search we can also adjust parameters with "main.py --bs 32 --lr 0.001" syntax.

Additionally, we will do greedy subject pooling, where a distance matrix is loaded, describing the distance between each subject based on one of three different distance metrics (also a HP). We iterate over each subject and for each subject we create a pool based on the shortest distance. First we train on only the main subjects data, then we add the clostest neighbours data to the dataset. If the validation loss decreases we keep the neighbour on the pool, if the validation loss increases we kick the neighbour out and take the next closes. After three failed attempts, we terminate the search for this main subject and move to the next one.
The algorithm I used for subject pooling looks something like this:

for i in range(number of subjects):
        # sub1 = i + 1
        for j in range(number of subjects):
            sub_list.append(distances.iloc[i,:].argmin() + 1)
            distances.iloc[i, distances.iloc[i,:].argmin()] = 1e5
            sub_list_str = '_'.join(str(x) for x in sub_list)

            #Call main.py
            #Get results
                        run_metrics = pd.read_csv(results_path)
            new_loss = float(run_metrics['loss'])

            print(old_loss)
            print(new_loss)
            if new_loss < old_loss:
                old_loss = new_loss
                grace_count = 0

            elif new_loss > old_loss:
                grace_count += 1
                sub_list = sub_list[:-1] ##Maybe bug because it deletes last item after 3 times fail? But i should still be in it 
            if grace_count == top or j == 10:# or distances.iloc[i,:].min() > 1e4:
                #
                best_combs.append(sub_list)
                old_loss = 1e5
                sub_list = []
                grace_count = 0
                break


The Subject pooling should always be done from scratch for each HP run.

The Results path look like this:
results/dataset/model_name/subject_{primary subject}/{HP_configuration}.zip
By iterating over the folder with .zip files you can get the validation loss and other metrics for analysis later.


This is how the .zip files are saved. I man want to change the HP I test for along the way, so please make that flexible.
base_folder = './results'
subject = f'subject_{sub}'

dropout = kwargs['dropout']
windows = kwargs['windows']
processor = kwargs['pre_processor']
encoder = kwargs['pre_encoder']
cut = kwargs['cutfill']
dec = kwargs['learn_decoder']
predec = kwargs['learn_predecoder']
lora = kwargs['learn_lora']
llr = kwargs['lora_lr']

hp_config = f'win{windows}_bs{bs}_lr{lr}_wd{wd}_dp{dropout}_llr{llr}_proc{processor}_enc{encoder}_cut{cut}_dec{dec}_predec{predec}_lora{lora}_sd{sd}'

model_folder = os.path.join(base_folder, dataset, model_name, subject, hp_config)

output_file = os.path.join(model_folder, f'run_{run_id}.csv')


Please also track ['train_losses', 'train_accs', 'val_losses', 'val_accs', 'test_losses', 'test_accs']. That is also how they are saved in the .zip files.

I also want to get standard deviations in the end, so each run needs an id. For those runs with the same hyperparameters, the subject pool should also be locked, so thats needs to be saved and loaded also.

---------------------------------------

Okay so I used your script and adapted some stuff to integrate it into the rest of my code and it seems to work good so far. The experiments are running but will take a long time to finish. I want to get started on an extended analysis script. 

In the end, the results metrics for the completed runs should be loaded from the csv files. The respective hyperparameter combination can be read from the folder name. After the pooling, the best pool does five training runs so we can compute the standard deviation over these five runs. So only the folders containing all five runs should be taken into consideration, the others are just test runs for the pooling procedure.
For each hyperparameter combination compute the mean and standard deviation for all tracked metrics over the five runs and display them in a table and save that table as a csv file. Also if the results do not exist yet, skip them 

results_path = os.path.join(BASE_FOLDER, DATASET, MODEL_NAME, subject_str, hp_config)
target_csv = os.path.join(results_path, f"run_{run_id}.csv")
BASE_FOLDER = "./results"
DATASET = "mamem" #or "bci" or "bcicha"
MODEL_NAME = "inception" #or BaselineDeviationModelIdEmbedHeadLora
RUN_IDENTIFIERS = [10, 11, 12, 13, 14]

These are some other important functions and Global parameters from my script:

HP_GRID = {
    'inception': {
        'bs': [64, 128], #32, 
        'lr': [1e-3, 1e-4], #1e-2, 
        'dist': ['AE', 'COV'], #, 'KL'
        'wd': [1e-2, 1e-3], 
        'filter_len': [40], #, 60
        'run_id': RUN_IDENTIFIERS
    },
    'BaselineDeviationModelIdEmbedHeadLora':{
        'bs': [64, 128],
        'lr': [1e-2, 1e-3, 1e-4],
        'dist': ['AE', 'COV', 'KL'],
        'wd': [1e-2, 1e-3],
        'learn_decoder': [0, 1],
        'scheduler': [0, 1],
        'run_id': RUN_IDENTIFIERS
    }
}

TRACKED_METRICS = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']
BCICHA_SUB_ID = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]

def add_default_params(hp):
    if MODEL_NAME == 'BaselineDeviationModelIdEmbedHeadLora':
        if DATASET == 'mamem':
            hp['filter_len'] = 0
            hp['windows'] = 1
            hp['dropout'] = 0
            hp['lora_lr'] = 1e-5
            hp['pre_processor'] = True
            hp['pre_encoder'] = True
            hp['cutfill'] = True
            hp['learn_predecoder'] = False
            hp['learn_lora'] = True
        if DATASET == 'bci':
            pass #Don't have params, also need to train recon model first
            # hp['filter_len'] = 0
            # hp['windows'] =  
            # hp['dropout'] =
            # hp['lora_lr'] =
            # hp['pre_processor'] =
            # hp['pre_encoder'] =
            # hp['cutfill'] =
            # hp['learn_predecoder'] =
            # hp['learn_lora'] =
        if DATASET == 'bcicha':
            hp['filter_len'] = 0
            hp['windows'] = 1
            hp['dropout'] = 0
            hp['lora_lr'] = 1e-1
            hp['pre_processor'] = True
            hp['pre_encoder'] = True
            hp['cutfill'] = True
            hp['learn_predecoder'] = False
            hp['learn_lora'] = True
    elif MODEL_NAME == 'inception':
        hp['learn_decoder'] = False
        hp['scheduler'] = True
        hp['windows'] = 0
        hp['dropout'] = 0.0
        hp['lora_lr'] = 0.0
        hp['pre_processor'] = False
        hp['pre_encoder'] = False
        hp['cutfill'] = False
        hp['learn_predecoder'] = False
        hp['learn_lora'] = False
    return hp

def get_hp_config_string(hp):
    """
    Generates the exact folder name based on custom naming convention.
    """
    #cache_path = WindowsPath('locked_pools/mamem/inception/sub_1_pool_bs32_lr0.01_distAE_wd0.01_filterlen40_learndecoderFalse_schedulerTrue_windows0_dropout0.0_loralr0.0_preprocessorFalse_preencoderFalse_cutfillFalse_learnpredecoderFalse_learnloraFalse.json'),
# DATASET = 'mamem',
# MODEL_NAME = 'inception'
    return (f"bs{hp['bs']}_lr{hp['lr']}_dist{hp['dist']}_wd{hp['wd']}_fl{hp[filter_len']}_dec{hp['learn_decoder']}_"
            f"sd{hp['scheduler']}_win{hp['windows']}_dp{hp['dropout']}_llr{hp['lora_lr']}_proc{hp['pre_processor']}_"
            f"enc{hp['pre_encoder']}_cut{hp['cutfill']}_predec{hp['learn_predecoder']}_lora{hp['learn_lora']}")

def get_base_hp_identifier(hp):
    """
    Generates a unique identifier for a configuration EXCLUDING the run identifier (id).
    Used to lock and cache the subject pool.
    """
    hp_copy = hp.copy()
    hp_copy.pop('run_id', None)
    return "_".join(f"{k.replace('_', '')}{v}" for k, v in hp_copy.items())