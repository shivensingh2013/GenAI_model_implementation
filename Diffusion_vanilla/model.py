import torch
import torch.nn as nn
import utils
from utils import NormActConv, TimeEmbedding,SelfAttentionBlock,Downsample,Upsample
from typing import List

"""Unet architecture 
Input : 1 ) Xt generated fro diffusion forward process (non trainable)
2) T time step in some transformed form

Output - E theta to be fed to diffusion reverse process"""


def get_time_embedding(time_steps:torch.Tensor, t_emb_dim:int) -> torch.Tensor :
        """ 
        Transform a scalar time-step into a vector representation of size t_emb_dim.
        
        :param time_steps: 1D tensor of size -> (Batch,)
        :param t_emb_dim: Embedding Dimension -> for ex: 128 (scalar value)
        
        :return tensor of size -> (B, t_emb_dim)
        """

        assert t_emb_dim%2 == 0, "time embedding must be divisible by 2."
        factor = 2 * torch.arange(start = 0, 
                              end = t_emb_dim//2, 
                              dtype=torch.float32, 
                              device=time_steps.device) / (t_emb_dim)
        factor = 10000**factor
        t_emb = time_steps[:,None]
        t_emb = t_emb/factor 
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)
        return t_emb 


"""Architecture of UNET
Down-Convolutional Block
Mid-Conv Block
Up-Conv block
"""

class DownC(nn.Module):
        """
    Perform Down-convolution on the input using following approach.
    1. Conv + TimeEmbedding
    2. Conv
    3. Skip-connection from input x.
    4. Self-Attention
    5. Skip-Connection from 3.
    6. Downsampling
    """
        def __init__(self,
                        in_channels:int,
                        out_channels:int,
                        t_emb_dim:int = 128,
                        num_layers:int = 2,
                        down_sample:bool = True):

                super(DownC,self).__init__()
                self.num_layers = num_layers
                self.conv1 = nn.ModuleList([
                utils.NormActConv(in_channels if i==0 else out_channels, 
                                out_channels
                        ) for i in range(num_layers)
                ])

                self.conv2 = nn.ModuleList([
                utils.NormActConv(out_channels, 
                                out_channels
                        ) for _ in range(num_layers)
                ])

                self.te_block = nn.ModuleList([
                utils.TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers)
                ])
                self.attn_block = nn.ModuleList([
                utils.SelfAttentionBlock(out_channels) for _ in range(num_layers)
                ])

                self.down_block =utils.Downsample(out_channels, out_channels) if down_sample else nn.Identity()
                self.res_block = nn.ModuleList([
                nn.Conv2d(
                        in_channels if i==0 else out_channels, 
                        out_channels, 
                        kernel_size=1
                ) for i in range(num_layers)
                ])

        def forward(self,x,t_emb):
                out = x
                for i in range(self.num_layers):
                        resnet_input = out
                        out = self.conv1[i](out)
                        out = out + self.te_block[i](t_emb)[:, :, None, None]
                        out = self.conv2[i](out)
                        out = out + self.res_block[i](resnet_input)
                        out_attn = self.attn_block[i](out)
                        out = out + out_attn
                # Downsampling
                out = self.down_block(out)
                return out

class MidC(nn.Module):
        """
                Refine the features obtained from the DownC block.
                It refines the features using following operations:

                1. Resnet Block with Time Embedding
                2. A Series of Self-Attention + Resnet Block with Time-Embedding 
        """
        def __init__(self,
                in_channels:int, 
                out_channels:int,
                t_emb_dim:int = 128,
                num_layers:int = 2
                ):
                super(MidC, self).__init__()
                
                self.num_layers = num_layers
                self.conv1 = nn.ModuleList([
                utils.NormActConv(in_channels if i==0 else out_channels, 
                                out_channels
                        ) for i in range(num_layers + 1)
                ])
                self.conv2 = nn.ModuleList([
                utils.NormActConv(out_channels, 
                        out_channels
                       ) for _ in range(num_layers + 1)
                ])
                self.te_block = nn.ModuleList([
                utils.TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers + 1)
                ])
                self.attn_block = nn.ModuleList([
                utils.SelfAttentionBlock(out_channels) for _ in range(num_layers)
                ])

                self.res_block = nn.ModuleList([
                nn.Conv2d(
                in_channels if i==0 else out_channels, 
                out_channels, 
                kernel_size=1
                ) for i in range(num_layers + 1)
                ])

        def forward(self,x,t_emb):
                out = x
                 # First-Resnet Block
                resnet_input = out
                out = self.conv1[0](out)
                out = out + self.te_block[0](t_emb)[:, :, None, None]
                out = self.conv2[0](out)
                out = out + self.res_block[0](resnet_input)

                # sequence of SA_resnet block
                for i in range(self.num_layers):
                        out_attn = self.attn_block[i](out)
                        out = out + out_attn

                        resnet_input = out
                        out = self.conv1[i+1](out)
                        out = out + self.te_block[i+1](t_emb)[:, :, None, None]
                        out = self.conv2[i+1](out)
                        out = out + self.res_block[i+1](resnet_input)
                return out

class UpC(nn.Module):
    """
    Perform Up-convolution on the input using following approach.
    1. Upsampling
    2. Conv + TimeEmbedding
    3. Conv
    4. Skip-connection from 1.
    5. Self-Attention
    6. Skip-Connection from 3.
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 t_emb_dim:int = 128, # Time Embedding Dimension
                 num_layers:int = 2,
                 up_sample:bool = True # True for Upsampling
                ):
        super(UpC, self).__init__()
        
        self.num_layers = num_layers
        
        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i==0 else out_channels, 
                        out_channels
                       ) for i in range(num_layers)
        ])
        
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, 
                        out_channels
                       ) for _ in range(num_layers)
        ])
        
        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers)
        ])
        
        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) for _ in range(num_layers)
        ])
        
        self.up_block =Upsample(in_channels, in_channels//2) if up_sample else nn.Identity()
        
        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i==0 else out_channels, 
                out_channels, 
                kernel_size=1
            ) for i in range(num_layers)
        ])
        
    def forward(self, x, down_out, t_emb):
        
        # Upsampling
        x = self.up_block(x)
        x = torch.cat([x, down_out], dim=1)
        
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            
            # Resnet Block
            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)

            # Self Attention
            out_attn = self.attn_block[i](out)
            out = out + out_attn
        
        return out

class Unet(nn.Module):
        
        """
                U-net architecture which is used to predict noise
                in the paper "Denoising Diffusion Probabilistic Model".

                U-net consists of Series of DownC blocks followed by MidC
                followed by UpC.
        """ 
        def __init__(self,
                 im_channels: int = 1, # RGB 
                 down_ch: list = [32, 64, 128, 256],
                 mid_ch: list = [256, 256, 128],
                 up_ch: List[int] = [256, 128, 64, 16],
                 down_sample: List[bool] = [True, True, False],
                 t_emb_dim: int = 128,
                 num_downc_layers:int = 2, 
                 num_midc_layers:int = 2, 
                 num_upc_layers:int = 2
                ):
                super(Unet, self).__init__()
                self.im_channels = im_channels
                self.down_ch = down_ch
                self.mid_ch = mid_ch
                self.up_ch = up_ch
                self.t_emb_dim = t_emb_dim
                self.down_sample = down_sample
                self.num_downc_layers = num_downc_layers
                self.num_midc_layers = num_midc_layers
                self.num_upc_layers = num_upc_layers
                self.up_sample = list(reversed(self.down_sample)) # [False, True, True]
                ## converts from 3 channel image to 32 channel convolved latent
                self.cv1 = nn.Conv2d(self.im_channels, self.down_ch[0], kernel_size=3, padding=1)
                # Initial Time Embedding Projection
                self.t_proj = nn.Sequential(
                nn.Linear(self.t_emb_dim, self.t_emb_dim), 
                nn.SiLU(), 
                nn.Linear(self.t_emb_dim, self.t_emb_dim)
                )
                # DownC Blocks
                self.downs = nn.ModuleList([
                DownC(
                        self.down_ch[i], 
                        self.down_ch[i+1], 
                        self.t_emb_dim, 
                        self.num_downc_layers, 
                        self.down_sample[i]
                ) for i in range(len(self.down_ch) - 1)
                ])
                # MidC Block
                self.mids = nn.ModuleList([
                MidC(
                        self.mid_ch[i], 
                        self.mid_ch[i+1], 
                        self.t_emb_dim, 
                        self.num_midc_layers
                ) for i in range(len(self.mid_ch) - 1)
                ])

                # UpC Block
                self.ups = nn.ModuleList([
                UpC(
                        self.up_ch[i], 
                        self.up_ch[i+1], 
                        self.t_emb_dim, 
                        self.num_upc_layers, 
                        self.up_sample[i]
                ) for i in range(len(self.up_ch) - 1)
                ])

                ## takes input as the last output of UP and changes into  a RGB image
                self.cv2 = nn.Sequential(
                        nn.GroupNorm(8, self.up_ch[-1]), 
                        nn.Conv2d(self.up_ch[-1], self.im_channels, kernel_size=3, padding=1)
                )


        def forward(self, x, t):
                out = self.cv1(x)
                
                # Time Projection
                t_emb = get_time_embedding(t, self.t_emb_dim)
                t_emb = self.t_proj(t_emb)
                
                # DownC outputs
                down_outs = []
                
                for down in self.downs:
                        down_outs.append(out)
                        out = down(out, t_emb)
                
                # MidC outputs
                for mid in self.mids:
                        out = mid(out, t_emb)
                
                # UpC Blocks
                for up in self.ups:
                        down_out = down_outs.pop()
                        out = up(out, down_out, t_emb)
                
                # Final Conv
                out = self.cv2(out)
                
                return out

if __name__ == "__main__":
        # Test
        model = Unet()
        x = torch.randn(4, 1, 32, 32)
        t = torch.randint(0, 10, (4,))
        print(model(x, t).shape)






