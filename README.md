# Unconditional_diffusion_image_training
## Most of this is tooken from the notebook

🤗 Training with Diffusers

In recent months, it has become clear that diffusion models have taken the throne as the "state-of-the-art" generative models. This code is a modified version of a Official Hugging Face code. This is Made to use Hugging Face's brand new Diffusers library to train a simple diffusion model.

This notebook leverages the 🤗 Datasets library to load and preprocess image datasets and the 🤗 Accelerate library to simplify training on any number of GPUs, with features like automatic gradient accumulation and tensorboard logging.

What i have Edited and removed Notabley is the Augmentation Switching that will cause problamatic training in small datasets making it learn to make some elements backwards this is meant to learn the raw true image for what it is.

# How training works
Here we set up our diffusion model. Diffusion models are neural networks that are trained to predict slightly less noisy images from a noisy input. At inference, they can be used to iteratively transform a random noise to generate an image:
![image](https://user-images.githubusercontent.com/10695622/174349667-04e9e485-793b-429a-affe-096e8199ad5b.png)
