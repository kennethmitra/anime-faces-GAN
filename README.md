This project was completed by Jacob Bunch, William Gu, Rakesh Johny, Kenneth Mitra, Andrea Nguyen, Alex Stahl, and Amitesh Yeleswarapu. In this article, we will outline our approach of creating anime style characters similar to real human faces using generative adversarial networks (GANs).

## Dataset Collection

To train our GAN we obtained a collection of anime-styled images from several sources. From kitsu.io we downloaded images from their dataset using curl. We found additional datasets from kaggle and a github user who scraped danbooru.donmai.us and uploaded his/her images to a google drive folder. You can find a list of links to our sources below:

- https://kitsu.io/explore/anime
- https://drive.google.com/file/d/0B4wZXrs0DHMHMEl1ODVpMjRTWEk/view?usp=sharing
- https://www.kaggle.com/soumikrakshit/anime-faces

After downloading these images to our local machine, we needed to extract only faces from the images. To process these images we ran a script that used OpenCV and an anime face detector, [Cascade Classifier](https://github.com/nagadomi/lbpcascade_animeface), to find a face and crop the image to a 256x256 PNG.

## Training the DCGAN
For our GAN, we used a DCGAN architecture since it was the easiest to train. The Generator was made up of four blocks of transpose convolution followed by batch normalization and ReLU activation. The last part of the network used a transpose convolution followed by a Tanh activation function to scale the outputs between -1 and 1. The Discriminator was made of blocks of convolution followed by batch normalization and a leaky ReLU activation. The output layers of the discriminator were made of a convolution followed by a sigmoid activation to output binary class probabilities of the input image being real or fake.

![GAN Model Diagram](/readme_images/GAN_model.png "GAN Model Diagram")

_Outline of GAN architecture_

Initially, we trained our DCGAN on a dataset of frames taken from a Minecraft video on YouTube. However, we found that although the results looked promising at first, the GAN soon fell into a state of mode collapse and the output images began to look the same.

![GAN trained on Minecraft screenshots](/readme_images/GAN_minecraft.png "GAN trained on Minecraft screenshots")

_Results of GAN trained on frames of Minecraft video after 73000 gradient descent steps. Most images are repeated, indicating mode collapse._

## Preventing Mode Collapse
We then decided to train our GAN to generate anime faces. In order to prevent mode collapse, we made a few changes to the training process.
### Two-Time Update Rule
In order to prevent the generator from overcompensating for the flaws the discriminator finds in the generated images (to the detriment of the rest of the image), we lowered the generator’s learning rate to 0.0001 and raised the discriminator’s learning rate to 0.0004. This has the effect of forcing the generator to generalize better.
### Increased Batch Size
According to [this paper](https://arxiv.org/pdf/1809.11096.pdf) on BigGAN, increased batch sizes led to an improvement in the FID and Inception Scores up to a point. In order to improve our GAN’s performance, we increased the batch size to 512 images.
### One-sided Label Smoothing
According to [this article](https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628), one-sided label smoothing can improve the performance of a GAN by making the discriminator less confident about an image being real. We applied one-sided label smoothing by lowering the label for real images from 1.0 to 0.9.
### Discriminator Noise
The same article also suggested adding noise to the discriminator inputs to improve training. Adding noise has the effect of making the discriminator generalize better instead of relying on a few small features to make the real/fake distinction. To this end, we added Gaussian noise with a mean of 0 and a standard deviation of 0.02 to the input of our discriminator.

## Training Progress
Below are some images of our model’s progress after certain numbers of steps of gradient descent.

![GAN output after 600 epochs](/readme_images/GAN_600.png "GAN output after 600 epochs")

_After 600 steps of gradient descent_

![GAN output after 2400 epochs](/readme_images/GAN_2400.png "GAN output after 2400 epochs")

_After 2400 steps of gradient descent_

![GAN output after 20000 epochs](/readme_images/GAN_20000.png "GAN output after 20000 epochs")

_After 20,000 steps of gradient descent_

![GAN output after 57300 epochs](/readme_images/GAN_57300.png "GAN output after 57300 epochs")

_After 57,300 steps of gradient descent_

## Latent Optimization
In order to generate images similar to human faces, we attempted to optimize the latent z-vectors of the Generative Network through stochastic gradient descent. The target image was normalized and rescaled to [-1,1] to match the output of the DCGAN. Different loss functions were then implemented to train the output of the GAN to imitate the target image:

### MSE
The mean squared error between the pixels of the generated image and the target image was a useful measure of the similarity between the images, however, this metric does not account for the perceived change in the structure of the changes
### LPIPS
The Learned Perceptual Image Patch Similarity metric uses a learned similarity model to evaluate the perceived similarity between the target image and the generated image. This metric ensures that the generated images looked close to realistic compared to the target image
### MSSSIM
The Multi Scale Structural Similarity Index also calculates the structural similarity between the two images. Using this metric as a loss function (1 — MS SSIM(target, generated)) helped increase the similarity in structure between the images.
### Edge Detection + MSE
The images were converted to grayscale and a Prewitt filter was used to find the magnitude of the gradients in the target and the GAN generated images. The mean squared error between the edge map of the target and the edge map of the GAN generated image is also computed.
### Edge Detection + LPIPS
The edge maps of the target and GAN generated images are compared as before but LPIPS is used to find the distance between the two.

Using these loss functions, we ran gradient descent using the Adam optimizer (lr=0.1) for 10000 steps.

## Inverting GAN Generated Images
First, to make sure our gradient descent search was working as expected, we attempted to invert an image that had been generated by the GAN. To make sure we explored all of our options, we experimented with several different loss functions:

![Target Image](/readme_images/inverting_target.png "Target Image")

_Target Image_

![Result with MSE Loss](/readme_images/inverting_mse.png "Result with MSE Loss")

_Result with MSE Loss_

![Result with LPIPS Loss](/readme_images/inverting_lpips.png "Result with LPIPS Loss")

_Result with LPIPS Loss_

![Result with Edge Detection + MSE Loss](/readme_images/inverting_edge_mse.png "Result with Edge Detection + MSE Loss")

_Result with Edge Detection + MSE Loss_

![Result with Edge Detection + LPIPS Loss](/readme_images/inverting_edge_lpips.png "Result with Edge Detection + LPIPS Loss")

_Result with Edge Detection + LPIPS Loss_

As you can see, the LPIPS and MSE loss metrics performed the best. Out of the edge detection loss functions, the Edge Detection + LPIPS performed better than the Edge Detection + MSE loss. Even though the colors in the image are very different from the target, the general shape of the face and the hair in between the eyes is preserved with edge detection.

## Inverting Anime Faces
The next step was to invert an anime face that the GAN had never seen in training. We once again compared the performance of several loss functions.

![Target Anime Face](/readme_images/invertin_anime_target.png "Target Anime Face")

_Target Anime Face_

![Result with MSE Loss](/readme_images/inverting_anime_mse.png "Result with MSE Loss")

_Result with MSE Loss_

![Result with LPIPS Loss](/readme_images/inverting_anime_lpips.png "Result with LPIPS Loss")

_Result with LPIPS Loss_

![Result with Edge + MSE Loss](/readme_images/inverting_anime_edge_mse.png "Result with Edge Detection + MSE Loss")

_Result with Edge Detection + MSE Loss_

![Result with Edge Detection + LPIPS Loss](/readme_images/inverting_anime_edge_lpips.png "Result with Edge Detection + LPIPS Loss")

_Result with Edge Detection + LPIPS Loss_

![Result with LPIPS + MSSSIM Loss](/readme_images/inverting_anime_lpips_msssim.png "Result with LPIPS + MSSSIM Loss")

_Result with LPIPS + MSSSIM Loss_

![Result with MSSSIM Loss](/readme_images/inverting_anime_msssim.png "Result with MSSSIM Loss")

_Result with MSSSIM Loss_

We can see that LPIPS performed the best in this case as well. The LPIPS image looks better overall, and the nose and mouth shapes are better preserved. The MSE Edge Detection produced unsatisfactory results but the LPIPS Edge Detection produces an image with reasonably similar edges as the target.

## Inverting Human Faces
We then moved on to finding the closest human face to a given anime face. While experimenting with different images, we noticed that the position of the face in the target image had a big effect on the resulting output.

![Target Image](/readme_images/inverting_human_target.jpg "Target Image")

_Target Image_

![Result with MSSSIM Loss](/readme_images/inverting_anime_msssim.png "Result with MSSSIM Loss")

_Target: shows face and shoulders_

![Result with LPIPS Loss](/readme_images/inverting_human_lpips.png "Result with LPIPS Loss")

_Result with LPIPS Loss_

![Result with MSE Loss](/readme_images/inverting_human_mse.png "Result with MSE Loss")

_Result with MSE Loss_

![Result with MSSSIM + LPIPS Result Loss](/readme_images/inverting_human_msssim_lpips.png "Result with Edge Detection + LPIPS Loss")

_Result with MSSSIM + LPIPS Loss_

![Result with Weighted Combination Loss](/readme_images/inverting_human_weighted.png "Result with Weighted Combination of Losses")

_Result with Weighted Combination of Losses_

Now, notice that if we shift the face in the target image down slightly, we get the following:

![Shifted Target Image](/readme_images/inverting_human_shifted_target.jpg "Shifted Target Image")

_Shifted Target Image_

![Result with LPIPS Loss](/readme_images/inverting_human_shifted_lpips.png "Result with LPIPS Loss")

_Result with LPIPS Loss_

![Result with MSE Loss](/readme_images/inverting_human_shifted_mse.png "Result with MSE Loss")

_Result with MSE Loss_

You can see that the way the target image is cropped plays a big role in the performance of the latent space search.

Based on this, we had the idea of having the latent space search also search for the best way to translate, rotate, and crop the target image. In order to accomplish this, we used PyTorch’s affine_grid() and grid_sample() functions to transform the target image in a differentiable way.We then defined the affine loss as the sum of LPIPS(<transformed target image>, <GAN generated image>) + SUM of the square of all affine parameters. This affine loss was added to the main loss function to get the overall loss.

We can see the comparison between LPIPS only and LPIPS + our affine transformation loss below:
  
![Result with LPIPS Loss Only](/readme_images/inverting_human_autoshift_lpips.png "Result with LPIPS Loss Only")

_Result with LPIPS Loss Only_
  
![Result with LPIPS + Affine Transformation Loss](/readme_images/inverting_human_autoshift_lpips_affine.png "Result with LPIPS + Affine Transformation Loss")

_Result with LPIPS + Affine Transformation Loss_
  
We can see in the images above that the latent optimization shifts the target image to the right to center the face and make it more like the images in the training dataset. However, the resulting image does not look as great, possibly due to the affine transformations making the loss function’s topology much more complex. However, while the image without the affine transformation looks more like an anime character, the image resulting from the affine transformation + LPIPS loss more closely resembles the target human image.
  
## Image Completion
To ensure that the missing region generated has a similar context to the non-missing region, a binary mask is used. A 0 corresponds to the corrupted region of the image and 1 corresponds to the uncorrupted region in the image. The product of the generated image and the mask was then used for the loss function in the latent optimization for Image Completion.

![Target Image](/readme_images/completion_animeInDataset_target.png "Target Image")

_Target Image (included in training dataset)_
  
![Result](/readme_images/completion_animeInDataset_result.png "Result")

_Retrieved Image_
  
![Target Image](/readme_images/completion_animeNotInDataset_target.png "Target Image")

_Target Image (not seen in training dataset)_
  
![Result](/readme_images/completion_animeNotInDataset_result.png "Result")

_Retrieved Image_
  
![Target Image](/readme_images/completion_human_target.png "Target Image")

_Human Face Target Image_
  
![Result](/readme_images/completion_human_msssim.png "Result with MSSSIM")

_Result with masked MS-SSIM Loss_
  
![Result](/readme_images/completion_human_LPIPS.png "Result with LPIPS")

_Result with masked LPIPS Loss_
  
We can see that the model was successful with censored images from the training dataset. Animated images not seen during training were also retrieved, and these results were not as good inversion using the uncensored version of the same image, as shown previously. Images retrieved from censored human images were inaccurate still, with MSSSIM having greater structural similarity and LPIPS having more realistic looking images.
  
## Super Resolution
Since we didn’t have the time to train our GAN on images larger than 64x64, we also looked into training a model to upscale our images. We started with an ordered dataset of high-resolution (256x256) anime faces and downscaled them to two sizes, 64x64 and 128x128. Now that we had pairs of high and low resolution images, we were able to train a model using gradient descent and a convolutional neural network in order to upscale the images. The architecture of our model is shown in the image below:

![Super Resolution Model Diagram](/readme_images/superres_diagram.png "Super Resolution Model Diagram")

_Super Resolution Model Diagram_
  
We began by using a bicubic up-sampling filter, and then we attempted to predict the mistakes of that filter in order to produce an image closer to the real high-resolution image. We trained the model with a learning rate of 0.001 and a batch size of 8 over 27177 gradient descent steps. Below we can see the results of the final trained super resolution model. Our model outperforms a simple bicubic up-sampling as can be seen in the crispness of the hair and mouth:

![Super Resolution Model Output](/readme_images/superres_output.png "Super Resolution Model Output")

## Latent Interpolation
To test the performance of the super resolution model and the generalization of our GAN, we tried a simple interpolation between two GAN generated images. We start with one image and take steps towards the goal image, passing each step through the super resolution model.

![Latent Interpolation Start Image](/readme_images/latent_interp_start.png "Latent Interpolation Start Image")
  
_Starting Image_
  
![Latent Interpolation End Image](/readme_images/latent_interp_end.png "Latent Interpolation End Image")

_Ending Image_
  


https://user-images.githubusercontent.com/14254758/130308760-fa3c1194-b3c3-45f9-9a3a-236fc905cad5.mp4
  
We see that the interpolation is smooth and produces reasonable anime faces in between the starting and ending images so we can conclude that our GAN is generalizing well and not merely memorizing the dataset. We also see that the super resolution model performs satisfactorily, although it distorts the image slightly towards the end.
