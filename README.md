# README


This repository is the official implementation of "LAtte: Hyperbolic Lorentz Attention for Cross-Subject EEG Classification". 


## Dataset
Download the datasets and unzip them to the folder 'data'. 
1. BCIC-IV-2a:
    https://www.bbci.de/competition/iv/
2. MAMEM-SSVEP-II:
   https://www.mamem.eu/results/datasets/
3. BCI-ERN:
    https://www.kaggle.com/competitions/inria-bci-challenge/data

The download link is provided in download.md or follow this Link [data](https://drive.google.com/file/d/1RxN2PWOkYJw-NzyM0vaxdLkina2q-_Rj/view) to the download directly.

## Pretraining 
To pretrain the models, run the pretrain.py file. For example, to pretrain the Latte model on the BCIC-IV-2a dataset, 
run the following command:
python recon_main.py --model BaselineBlockModelIdEmbedLora --dataset mamem --sub all
<!-- python recon_main.py --model Latte --dataset bci --sub 'all' -->
The checkpoints used for the paper evaluation are in /reconstruction/checkpoints.

## Training
To train the models, run the main.py file. For example, to train the Latte model on the BCIC-IV-2a dataset,
run the following command:
python main.py --model Latte --dataset bci --sub 'all'

For the finetune set the --sub to the ID of the subject and --finetuning to True



## Reference
We modified code from:
```bash

@article{burchert2024eeg,
  title={Are EEG Sequences Time Series? EEG Classification with Time Series Models and Joint Subject Training},
  author={Burchert, Johannes and Werner, Thorben and Yalavarthi, Vijaya Krishna and de Portugal, Diego Coello and Stubbemann, Maximilian and Schmidt-Thieme, Lars},
  journal={arXiv preprint arXiv:2404.06966},
  year={2024}
}
```

```bash

@article{pan2022matt,
  title={MAtt: a manifold attention network for EEG decoding},
  author={Pan, Yue-Ting and Chou, Jing-Lun and Wei, Chun-Shu},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={31116--31129},
  year={2022}
}
```

```bash
@inproceedings{wei2019spatial,
  title={Spatial component-wise convolutional network (SCCNet) for motor-imagery EEG classification},
  author={Wei, Chun-Shu and Koike-Akino, Toshiaki and Wang, Ye},
  booktitle={2019 9th International IEEE/EMBS Conference on Neural Engineering (NER)},
  pages={328--331},
  year={2019},
  organization={IEEE}
}
```


```bash
@inproceedings{huang2017riemannian,
  title={A riemannian network for spd matrix learning},
  author={Huang, Zhiwu and Van Gool, Luc},
  booktitle={Thirty-first AAAI conference on artificial intelligence},
  year={2017}
}
```


```bash
@article{brunner2008bci,
  title={{BCI Competition 2008--Graz data set A}},
  author={Brunner, Clemens and Leeb, Robert and M{\"u}ller-Putz, Gernot and Schl{\"o}gl, Alois and Pfurtscheller, Gert},
  journal={Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces), Graz University of Technology},
  volume={16},
  pages={1--6},
  year={2008}
}
```


```bash
@article{Nikolopoulos2021,
author = "Spiros Nikolopoulos",
title = "{MAMEM EEG SSVEP Dataset II (256 channels, 11 subjects, 5 frequencies presented simultaneously)}",
year = "2021",
month = "5",
url = "https://figshare.com/articles/dataset/MAMEM_EEG_SSVEP_Dataset_II_256_channels_11_subjects_5_frequencies_presented_simultaneously_/3153409",
doi = "10.6084/m9.figshare.3153409.v4"
}

```

```bash
@article{margaux2012objective,
  title={{Objective and subjective evaluation of online error correction during P300-based spelling}},
  author={Margaux, Perrin and Emmanuel, Maby and S{\'e}bastien, Daligault and Olivier, Bertrand and J{\'e}r{\'e}mie, Mattout},
  journal={Advances in Human-Computer Interaction},
  volume={2012},
  year={2012},
  publisher={Hindawi}
}
```

```bash
@misc{spdnet2020,
  author={adavoudi},
  title={spdnet},
  year={2020},
  url={https://github.com/adavoudi/spdnet},
}
```

