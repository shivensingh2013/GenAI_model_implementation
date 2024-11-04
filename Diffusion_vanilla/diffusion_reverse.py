import torch
class DiffusionReverseProcess:

        """ this implements the reverse process . In forward process we computed noisy image at time t - > xt
        Now, we will see how we can use a UNET to predict -E_theta and generate x_0 from x_t using it.

        
        """
        def __init__(self,
                            num_time_steps = 1000,
                            beta_start = 1e-4,
                            beta_end = 0.02):
            self.b = torch.linspace(beta_start, beta_end, num_time_steps)
            self.a = 1 - self.b # a -> alpha
            self.a_bar = torch.cumprod(self.a, dim=0) # a_bar = alpha_bar
        
        ## compute x0 (real image from xt and etheta coming from UNET
        def sample_prev_timestep(self,xt,noise_pred,t):
            x0 = xt - (torch.sqrt(1 - self.a_bar.to(xt.device)[t]) * noise_pred)
            x0 = x0/torch.sqrt(self.a_bar.to(xt.device)[t])
            x0 = torch.clamp(x0, -1., 1.) 

            ## mean of x t-1
            mean = (xt - ((1 - self.a.to(xt.device)[t]) * noise_pred)/(torch.sqrt(1 - self.a_bar.to(xt.device)[t])))
            mean = mean/(torch.sqrt(self.a.to(xt.device)[t]))

            if t == 0:
                return mean, x0
            else:
                variance =  (1 - self.a_bar.to(xt.device)[t-1])/(1 - self.a_bar.to(xt.device)[t])
                variance = variance * self.b.to(xt.device)[t]
                sigma = variance**0.5
                z = torch.randn(xt.shape).to(xt.device)
                return mean + sigma * z, x0
    
if __name__ == "__main__":
    original = torch.randn(1, 1, 28, 28)
    ## this is the output from Unet to be used
    noise_pred = torch.randn(1, 1, 28, 28)
    t = torch.randint(0, 1000, (1,)) 

    # reverse Process
    drp = DiffusionReverseProcess()
    out, x0 = drp.sample_prev_timestep(original, noise_pred, t)
    out.shape



