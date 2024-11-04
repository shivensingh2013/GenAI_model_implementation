import torch
from PIL import Image
from torchvision import transforms


class DiffusionForward:

    """ implements xt = a summation of x0 and epsilon noise"""

    def __init__(self,
    num_time_steps = 1000,
    beta_start = 1e-4,
    beta_end = 0.02):
    ## array of 1000 betas
        self.betas = torch.linspace(beta_start,beta_end,num_time_steps)
        self.alphas = 1-self.betas
        self.alpha_bars  = torch.cumprod(self.alphas,dim = 0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1-self.alpha_bars)

    def add_noise(self,original,noise,t):
        """ Adds noise to a batch of original images at time-step t.

        :param original: Input Image Tensor
        :param noise: Random Noise Tensor sampled from Normal Dist N(0, 1)
        :param t: timestep of the forward process of shape -> (B, )

        Note: time-step t may differ for each image inside the batch.

        returns -  Xt - noise image as a combination of original image and noise weighted based on timestep
        """
        sqrt_alpha_bar_t = self.sqrt_alpha_bars.to(original.device)[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars.to(original.device)[t]

        ## done to match the dim of original image
        sqrt_alpha_bar_t = sqrt_alpha_bar_t[:, None, None, None]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t[:, None, None, None]
        xt = (sqrt_alpha_bar_t * original)+(sqrt_one_minus_alpha_bar_t * noise)
        assert original.shape == noise.shape , "Image and noise shape not matching"
        return xt

## unit testing the component
if __name__== "__main__" :
    obj = DiffusionForward()
    orig = Image.open(r"C:\Users\IHG6KOR\Desktop\shiv\Portfolio\shivensingh2013.github.io\P6_stanford_cs236\GenAI_model_implementation\Diffusion_vanilla\sample.png")
    device = torch.device(0)
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(orig).unsqueeze(0).to(device)
    # print(tensor.shape)
    noise = torch.randn(tensor.shape).to(device)
    t_steps = torch.randint(0, 10, (1,)) 
    noise_out_image = obj.add_noise(tensor,noise,t_steps)
    print(noise_out_image.shape)
    transformed_img = transforms.ToPILImage()(noise_out_image[0])
    transformed_img.show()






