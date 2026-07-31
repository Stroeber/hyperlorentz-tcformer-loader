import os
import numpy as np
import pandas as pd
import csv

def save_output(model_name, dataset, sub, bs, lr, wd, sd, alpha, epochs, results, **kwargs):

    base_folder = './results'
    subject = f"subject_{kwargs['sub_str']}"

    dropout = kwargs['dropout']
    windows = kwargs['windows']
    processor = kwargs['pre_processor']
    encoder = kwargs['pre_encoder']
    cut = kwargs['cutfill']
    dec = kwargs['learn_decoder']
    predec = kwargs['learn_predecoder']
    lora = kwargs['learn_lora']
    llr = kwargs['lora_lr']

    hp_config = (f"bs{bs}_lr{lr}_dist{kwargs['dist']}_wd{wd}_fl{kwargs['filter_len']}_dec{kwargs['learn_decoder']}_"
            f"sd{sd}_win{kwargs['windows']}_dp{kwargs['dropout']}_llr{kwargs['lora_lr']}_proc{kwargs['pre_processor']}_"
            f"enc{kwargs['pre_encoder']}_cut{kwargs['cutfill']}_predec{kwargs['learn_predecoder']}_lora{kwargs['learn_lora']}")
    if model_name == "mixer":
        hp_config += f"_blocks{kwargs['blocks']}_ffdim{kwargs['ff_dim']}_dropout{kwargs['dropout']}"
    elif model_name == "vit":
        hp_config += f"_depth{kwargs['depth']}_mlpdim{kwargs['mlp_dim']}_dropout{kwargs['dropout']}"

    model_folder = os.path.join(base_folder, dataset, model_name, hp_config, subject)####################
    os.makedirs(model_folder, exist_ok=True)

    # run_id = 1
    output_file = os.path.join(model_folder, f"run_{kwargs['run_id']}.csv")
    # while os.path.exists(output_file):
    #     run_id += 1
    #     output_file = os.path.join(model_folder, f'run_{run_id}.csv')

    outputs = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']

    # Flatten everything to 1D and convert to floats
    flat_results = [[float(x) for x in np.ravel(arr)] for arr in results]

    # Transpose so that each row is one epoch
    rows = list(zip(*flat_results))

    # Write CSV manually
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(outputs)  # header
        writer.writerows(rows)