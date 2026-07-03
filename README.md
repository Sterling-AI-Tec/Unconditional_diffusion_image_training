# Unconditional_diffusion_image_training
## Most of this is tooken from the notebook

🤗 Training with Diffusers

In recent months, it has become clear that diffusion models have taken the throne as the "state-of-the-art" generative models. This code is a modified version of a Official Hugging Face code. This is Made to use Hugging Face's brand new Diffusers library to train a simple diffusion model.

This notebook leverages the 🤗 Datasets library to load and preprocess image datasets and the 🤗 Accelerate library to simplify training on any number of GPUs, with features like automatic gradient accumulation and tensorboard logging.

What i have Edited and removed Notabley is the Augmentation Switching that will cause problamatic training in small datasets making it learn to make some elements backwards this is meant to learn the raw true image for what it is.

# How training works
Here we set up our diffusion model. Diffusion models are neural networks that are trained to predict slightly less noisy images from a noisy input. At inference, they can be used to iteratively transform a random noise to generate an image:
![image](https://user-images.githubusercontent.com/10695622/174349667-04e9e485-793b-429a-affe-096e8199ad5b.png)
Figure from DDPM paper (https://arxiv.org/abs/2006.11239). 

Don't worry too much about the math if you're not familiar with it, the import part to remember is that our model corresponds to the arrow pθ(xt−1|xt) (which is a fancy way of saying: predict a slightly less noisy image).

The interesting part is that it's really easy to add some noise to an image, so the training can happen in a semi-supervised fashion as follows:

  1. Take an image from the training set.
  2. Apply to it some random noise t times (this will give the xt−1 and the xt in the figure above).
  3. Give this noisy image to the model along with the value of t.
  4. Compute a loss from the output of the model and the noised image xt−1.

Then we can apply gradient descent and repeat this process multiple times.

Most diffusion models use architectures that are some variant of a U-net and that's What is used in this notebook.
![image]([https://us.aws.cdn.hf.co/xet-bridge-us/621ffdd236468d709f1835cf/2866a46eeef4b7383541cfc05f10c82de1a887f0d994ae9881e0df4cae10480d?response-content-type=image%2Fpng&response-content-disposition=inline%3B+filename*%3DUTF-8%27%27unet-model.png%3B+filename%3D%22unet-model.png%22%3B&X-Xet-Cas-Uid=public&user_id=public&Expires=1783062397&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly91cy5hd3MuY2RuLmhmLmNvL3hldC1icmlkZ2UtdXMvNjIxZmZkZDIzNjQ2OGQ3MDlmMTgzNWNmLzI4NjZhNDZlZWVmNGI3MzgzNTQxY2ZjMDVmMTBjODJkZTFhODg3ZjBkOTk0YWU5ODgxZTBkZjRjYWUxMDQ4MGRcXD9yZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcmcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj1pbmxpbmUlM0IrZmlsZW5hbWUlMkElM0RVVEYtOCUyNyUyN3VuZXQtbW9kZWwucG5nJTNCK2ZpbGVuYW1lJTNEJTIydW5ldC1tb2RlbC5wbmclMjIlM0ImWC1YZXQtQ2FzLVVpZD1wdWJsaWMmdXNlcl9pZD1wdWJsaWMiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkVwb2NoVGltZSI6MTc4MzA2MjM5N319fV19&Signature=MEUCIBTypCpSdLQL-ldKA8PIdOugHWemK4H5O-uMMzEMzGItAiEA9sVkD1PKk%7EMBII-dRqh1kIpRBJwGvL6C2run-wNoFGo_&Key-Pair-Id=01KAYHXK2CBJSW0YZTMNXK9W1M])

In a nutshell:

  •the model has the input image go through several blocks of ResNet layers which halves the image size by 2
  •then through the same number of blocks that upsample it again.
  •there are skip connections linking the features on the downample path to the corresponsding layers in the upsample path.

A key feature of this model is that it predicts images of the same size as the input, which is exactly what we need here.

Diffusers provides us a handy UNet2DModel class which creates the desired architecture in PyTorch.
