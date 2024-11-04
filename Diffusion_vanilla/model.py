import torch

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
        




