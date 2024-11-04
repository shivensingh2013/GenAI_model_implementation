import torch.nn as nn
import torch
""" extra utilities for UNET blocks
"""



class NormActConv(nn.Module):
    """
    Perform GroupNorm, Activation, and Convolution operations.
    """
    def __init__(self,
                            in_channels:int,
                            out_channels:int,
                             num_groups:int=8,
                             kernel_size:int = 3,
                             norm:bool= True,
                             act:bool = True):
        super(NormActConv,self).__init__()
        self.g_norm == nn.GroupNorm(num_groups,in_channels) if norm is True else nn.Identity()
        self.act = nn.SiLU() if act is True else nn.Identity()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,padding = (kernel_size - 1)//2 )
    
    def forward(self, x):
        x = self.g_norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

class TimeEmbedding(nn.Module):
     def __init__(self, 
                 n_out:int, # Output Dimension
                 t_emb_dim:int = 128 # Time Embedding Dimension
                ):
        super(TimeEmbedding, self).__init__()
        
        # Time Embedding Block
        self.te_block = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(t_emb_dim, n_out)
        )

        def forward(self, x):
            return self.te_block(x)



class SelfAttentionBlock(nn.Module):
    """
    Perform GroupNorm and Multiheaded Self Attention operation.    
    """
    def __init__(self, 
                 num_channels:int,
                 num_groups:int = 8, 
                 num_heads:int = 4,
                 norm:bool = True
                ):
        super(SelfAttentionBlock, self).__init__()

               # GroupNorm
        self.g_norm = nn.GroupNorm( num_groups,num_channels ) if norm is True else nn.Identity()

                #Attention
        self.attn = nn.MultiheadAttention(num_channels,num_heads,batch_first = True)

    def forward(self,x):
        batch_size, channels, h, w = x.shape
        x = x.reshape(batch_size, channels, h*w)
        x = self.g_norm(x)
        x = x.transpose(1, 2)
        x, _ = self.attn(x, x, x)
        ## output from attn - > 
        ## x- Attention output - Batch first- > N (number of sample) , L ( sequence length ) , E (embedding dimension)
        ## _ - Attention map - Batch First - > N (number of sample) , L (target sequence) , S (source sequence length)
        x = x.transpose(1, 2).reshape(batch_size, channels, h, w)
        return x



class Downsample(nn.Module):
    """
    Perform Downsampling by the factor of k across Height and Width.
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 k:int = 2, # Downsampling factor
                 use_conv:bool = True, # If Downsampling using conv-block
                 use_mpool:bool = True # If Downsampling using max-pool
                ):
        super(Downsample, self).__init__()
        self.use_conv = use_conv
        self.use_mpool = use_mpool

          # Downsampling using Convolution
        self.cv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), 
            nn.Conv2d(
                in_channels, 
                out_channels//2 if use_mpool else out_channels, 
                kernel_size=4, 
                stride=k, 
                padding=1
            )
        ) if use_conv else nn.Identity()

         # Downsampling using Maxpool
        self.mpool = nn.Sequential(
            nn.MaxPool2d(k, k), 
            nn.Conv2d(
                in_channels, 
                out_channels//2 if use_conv else out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0
            )
        ) if use_mpool else nn.Identity()

    def forward(self, x):
        
        if not self.use_conv:
            return self.mpool(x)
        
        if not self.use_mpool:
            return self.cv(x)
            
        return torch.cat([self.cv(x), self.mpool(x)], dim=1)


class Upsample(nn.Module):
    """
    Perform Upsampling by the factor of k across Height and Width
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 k:int = 2, # Upsampling factor
                 use_conv:bool = True, # Upsampling using conv-block
                 use_upsample:bool = True # Upsampling using nn.upsample
                ):
        super(Upsample, self).__init__()
        
        self.use_conv = use_conv
        self.use_upsample = use_upsample
        
        # Upsampling using conv
        self.cv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels//2 if use_upsample else out_channels, 
                kernel_size=4, 
                stride=k, 
                padding=1
            ),
            nn.Conv2d(
                out_channels//2 if use_upsample else out_channels, 
                out_channels//2 if use_upsample else out_channels, 
                kernel_size = 1, 
                stride=1, 
                padding=0
            )
        ) if use_conv else nn.Identity()
        
        # Upsamling using nn.Upsample
        self.up = nn.Sequential(
            nn.Upsample(
                scale_factor=k, 
                mode = 'bilinear', 
                align_corners=False
            ),
            nn.Conv2d(
                in_channels,
                out_channels//2 if use_conv else out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0
            )
        ) if use_upsample else nn.Identity()
        
    def forward(self, x):
        
        if not self.use_conv:
            return self.up(x)
        
        if not self.use_upsample:
            return self.cv(x)
        
        ## the one that is not used will pass a identity() which is same as input x but in right dimensions
        return torch.cat([self.cv(x), self.up(x)], dim=1)

if __name__ == "__main__":
    layer = Upsample(16, 32, 2, True, True)
    x = torch.randn(4, 16, 32, 32)
    print(f"{x.shape=}")
    print(f"{layer(x).shape=}")
