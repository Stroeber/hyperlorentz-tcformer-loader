bci_denoiser = {"in_channels": 1,
                "intermediate_channels": 22,
                "out_channels": 20,
                "kernel_timewise": 22,
                "kernel_channelwise": 12,
                "padding": 6,
                "num_subjects": 9}
mamem_denoiser = {"in_channels": 1,
                  "intermediate_channels": 125,
                  "out_channels": 15,
                  "kernel_timewise": 8,
                  "kernel_channelwise": 36,
                  "padding": 18,
                  "num_subjects": 11}
cha_denoiser = {"in_channels": 1,
                "intermediate_channels": 22,
                "out_channels": 16,
                "kernel_timewise": 56,
                "kernel_channelwise": 64,
                "padding": 32,
                "num_subjects": 16}

bci_encoder = {"inception_channels": 10,
               "curvature": 1,
               "learnable": True,
               "windows": 1,
               "sub_rank": 5,
               "intermediate": 1024,
               "lora_lr": 1e-1,
               "decoder_rank": 64}

mamem_encoder = {"inception_channels": 10,
                   "curvature": 0.8,
                   "learnable": True,
                   "windows": 1,
                   "sub_rank": 5,
                   "intermediate": 1024,
                   "lora_lr": 1e-4,
                   "decoder_rank": 64}

cha_encoder = {"inception_channels":10,
                   "curvature":3,
                   "learnable":True,
                   "windows":4,
                   "sub_rank":5,
                   "intermediate":1024,
                   "lora_lr":1e-1,
                   "decoder_rank":64}

DEFAULT_DENOISER_CONFIGS = {"bci": [bci_denoiser, 20],
                            "mamem": [mamem_denoiser, 15],
                            "bcicha": [cha_denoiser, 16]}

DEFAULT_MODEL_CONFIGS = {"bci": [bci_encoder],
                            "mamem": [mamem_encoder],
                            "bcicha": [cha_encoder]}

