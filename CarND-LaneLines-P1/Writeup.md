# **Finding Lane Lines on the Road in Image and Video data Writeup** 

## Objectives

The main objective is to create a software pipeline with certain helper functions to detect lane lines in image and video data. The following methods were implemented to obtain the final results:

1. Color selection to detect white and yellow lane lines
2. Convert image to grayscale for Canny Edge detection
3. Gaussian Blur to remove noise from the image
4. Canny Edge detection
5. Define a Region of Interest
6. Hough Transform
7. Average and extrapolate hough lines
8. Draw annotated lines on image/video data


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
