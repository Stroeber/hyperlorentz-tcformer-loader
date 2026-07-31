import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from models.criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder

class CBraMod(nn.Module):
    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, norm_first=True,
            activation=F.gelu
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)
        self.proj_out = nn.Sequential(
            # nn.Linear(d_model, d_model*2),
            # nn.GELU(),
            # nn.Linear(d_model*2, d_model),
            # nn.GELU(),
            nn.Linear(d_model, out_dim),
        )
        self.apply(_weights_init)

    def forward(self, x, mask=None, x_sub=None):
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)

        out = self.proj_out(feats)

        return out

class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                      groups=d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        # self.mask_encoding = nn.Parameter(torch.randn(in_dim), requires_grad=True)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
            # nn.LayerNorm(d_model, eps=1e-5),
        )
        # self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        # self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        # self.proj_in = nn.Sequential(
        #     nn.Linear(in_dim, d_model, bias=False),
        # )


    def forward(self, x, mask=None, x_sub=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        # print(f"Before time embedding, the dimension is {mask_x.shape}")

        patch_emb = self.proj_in(mask_x)
        # print(f"After time embedding, the dimension is {patch_emb.shape}")
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        mask_x = mask_x.contiguous().view(bz*ch_num*patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)
        # print(patch_emb[5, 5, 5, :])
        # print(spectral_emb[5, 5, 5, :])
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class CbraModHead(nn.Module):
    def __init__(self, param):
        super(CbraModHead, self).__init__()

        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )

        if param['use_pretrained_weights']:
            device = param['device']
            self.backbone.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth', map_location=device))
            print("Loaded Model parameters successfully")
        self.backbone.proj_out = nn.Identity()

        print('params',param)

        if param['dataset'] == 'bci':

            if param['classifier'] == 'avgpooling_patch_reps':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b d c s'),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(200, param['num_class']),
                )
            elif param['classifier'] == 'all_patch_reps_onelayer':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b (c s d)'),
                    nn.Linear(22 * 4 * 200, param.num_of_classes),
                )
            elif param['classifier'] == 'all_patch_reps_twolayer':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b (c s d)'),
                    nn.Linear(22 * 4 * 200, 200),
                    nn.ELU(),
                    nn.Dropout(param.dropout),
                    nn.Linear(200, param.num_of_classes),
                )
            elif param['classifier'] == 'all_patch_reps':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b (c s d)'),
                    nn.Linear(22 * 3 * 200, 4 * 200),
                    nn.ELU(),
                    nn.Dropout(param['dropout']),
                    nn.Linear(4 * 200, 200),
                    nn.ELU(),
                    nn.Dropout(param['dropout']),
                    nn.Linear(200, param['num_class']),
                )
        elif param['dataset'] == 'bcicha':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(56 * 200, 4 * 200),
                nn.ELU(),
                nn.Dropout(param['dropout']),
                nn.Linear(4 * 200, 200),
                nn.ELU(),
                nn.Dropout(param['dropout']),
                nn.Linear(200, param['num_class']),
            )
        elif param['dataset'] == 'mamem':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(8 * 200, 4 * 200),
                nn.ELU(),
                nn.Dropout(param['dropout']),
                nn.Linear(4 * 200, 200),
                nn.ELU(),
                nn.Dropout(param['dropout']),
                nn.Linear(200, param['num_class']),
            )


    def forward(self, x, mask=None, x_sub=None):
        # x = x / 100
        if x.dim() == 5:
            x = x.squeeze(1)

        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)  # (batch, channels, time)
        if x.dim() == 3:
            bz, ch, time = x.shape
            # Pad to multiple of patch_length
            remainder = time % 200
            if remainder != 0:
                pad_amount = 200 - remainder
                x = F.pad(x, (0, pad_amount), value=0)
            num_patches = x.shape[-1] // 200
            x = x.reshape(bz, ch, num_patches, 200)

        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out


def pass_through(X):
	return X


class InceptionBlock(nn.Module):
	def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU(), return_indices=False):
		"""
		: param in_channels				Number of input channels (input features)
		: param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
		: param kernel_sizes			List of kernel sizes for each convolution.
										Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
										This is nessesery because of padding size.
										For correction of kernel_sizes use function "correct_sizes".
		: param bottleneck_channels		Number of output channels in bottleneck.
										Bottleneck wont be used if nuber of in_channels is equal to 1.
		: param activation				Activation function for output tensor (nn.ReLU()).
		: param return_indices			Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d.
		"""
		super(InceptionBlock, self).__init__()
		self.return_indices=return_indices
		if in_channels > 1:
			self.bottleneck = nn.Conv1d(
								in_channels=in_channels,
								out_channels=bottleneck_channels,
								kernel_size=1,
								stride=1,
								bias=False
								)
		else:
			self.bottleneck = pass_through
			bottleneck_channels = 1
		self.conv_from_bottleneck_1 = nn.Conv1d(
										in_channels=bottleneck_channels,
										out_channels=n_filters,
										kernel_size=kernel_sizes[0],
										stride=1,
										padding=kernel_sizes[0]//2,
										bias=False
										)
		self.conv_from_bottleneck_2 = nn.Conv1d(
										in_channels=bottleneck_channels,
										out_channels=n_filters,
										kernel_size=kernel_sizes[1],
										stride=1,
										padding=kernel_sizes[1]//2,
										bias=False
										)
		self.conv_from_bottleneck_3 = nn.Conv1d(
										in_channels=bottleneck_channels,
										out_channels=n_filters,
										kernel_size=kernel_sizes[2],
										stride=1,
										padding=kernel_sizes[2]//2,
										bias=False
										)
		self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
		self.conv_from_max = nn.Conv1d(
									in_channels=in_channels,
									out_channels=n_filters,
									kernel_size=1,
									stride=1,
									padding=0,
									bias=False
									)
		self.batch_norm = nn.BatchNorm1d(num_features=4*n_filters)
		self.activation = activation

	def forward(self, X):
		# step 1
		Z_bottleneck = self.bottleneck(X)
		if self.return_indices:
			Z_maxpool, indices = self.max_pool(X)
		else:
			Z_maxpool = self.max_pool(X)
		# print(Z_bottleneck.shape)
		# step 2
		Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
		Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
		Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
		# print(Z1.shape)
		Z4 = self.conv_from_max(Z_maxpool)
		# step 3
		Z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
		Z = self.activation(self.batch_norm(Z))
		if self.return_indices:
			return Z, indices
		else:
			return Z

class InceptionModel(nn.Module):
    def __init__(self, in_channels, num_pred_classes, n_filters=32, kernel_sizes=[9, 19, 39],
                 bottleneck_channels=32, use_residual=True, activation=nn.ReLU(),
                 return_indices=False, multilayer_inception=True):
        super(InceptionModel, self).__init__()
        self.use_residual = use_residual
        self.return_indices = return_indices
        self.activation = activation
        self.multilayer_inception = multilayer_inception
        self.inception_1 = InceptionBlock(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_2 = InceptionBlock(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_3 = InceptionBlock(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=4 * n_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm1d(
                    num_features=4 * n_filters
                )
            )
        self.linear = nn.Linear(4 * n_filters, num_pred_classes)

    def forward(self, X):
        if self.multilayer_inception:
            Z = self.inception_1(X)
            Z = self.inception_2(Z)
            Z = self.inception_3(Z)
        else:
            Z = self.inception_1(X)
        Z = Z + self.residual(X)
        Z = self.activation(Z)
        Z = Z.mean(dim=-1)

        return self.linear(Z)

class CbramodInc(nn.Module):
    def __init__(self, encoder, decoder, freeze_encoder):
        super().__init__()
        self.encoder = encoder
        # Freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.decoder = decoder

    def forward(self, x, mask=None, x_sub=None):
        if x.dim() == 5:
            x = x.squeeze(1)
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)  # (batch, channels, time)
        if x.dim() == 3:
            bz, ch, time = x.shape
            # Pad to multiple of patch_length
            remainder = time % 200
            if remainder != 0:
                pad_amount = 200 - remainder
                x = F.pad(x, (0, pad_amount), value=0)
            num_patches = x.shape[-1] // 200
            x = x.reshape(bz, ch, num_patches, 200)
        feats = self.encoder(x)  # (batch, ch, patches, d_model)
        feats = torch.flatten(feats, start_dim=-2)  # (batch, ch, patches * d_model)
        return self.decoder(feats)

