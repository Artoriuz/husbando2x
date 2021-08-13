# husbando2x
[RCAN](https://arxiv.org/abs/1807.02758)-based SISR CNN for Japanese-style Anime/Manga/LN Art.

For more information please read my [blog post](https://artoriuz.github.io/blog/super_resolution_2.html) about it.

## Usage Instructions
This repository contains the necessary Python code to define, train and perform inference with the network. You'll need recent versions of TensorFlow and OpenCV for it to work.

## Included Weights
The included "model.h5" file can be used to load trained weights into the network. Training was done with 200 anime screenshots downscaled to 270x480 and 540x960 as the input/output pairs respectively.

These inputs look like this:
<img src="https://artoriuz.github.io/blog/images/husbando2x/trainingdata.png">

## Results
You can check some comparions between Waifu2x and Husbando2x in the image below:
<img src="https://artoriuz.github.io/blog/images/husbando2x/comparison2.png">
