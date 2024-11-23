class   CONFIG: 
        model_path = r'/home/ihg6kor/GenAI_model_implementation/Diffusion_vanilla/pretrained/ddpm_unet.pth'
        generated_csv_path = r'/home/ihg6kor/GenAI_model_implementation/dataset/mnist_generated_data.csv'
        train_csv_path= r"/home/ihg6kor/GenAI_model_implementation/dataset/fashion-mnist_train.csv"
        test_csv_path = r"/home/ihg6kor/GenAI_model_implementation/dataset/fashion-mnist_test.csv"
        num_epochs = 10
        lr = 1e-4
        num_timesteps = 1000
        batch_size = 32
        img_size = 28
        in_channels = 1
        num_img_to_generate = 8