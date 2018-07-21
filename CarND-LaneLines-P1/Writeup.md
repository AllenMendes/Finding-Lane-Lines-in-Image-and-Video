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
8. Draw annotated lines on image and video data

## Reflection

### 1. Software Pipeline
---
The flow of the software piepline is explained in the following sections along with the detailed explaination of each helper function and the reasons for selecting various parameters in them:

  #### 1. Color selection to detect white and yellow lane lines
  After reading an image, the first task is to robustly detect the white and yellow lines in an image. 
  ##### Original image
  ![challenge_image1](https://user-images.githubusercontent.com/8627486/43037545-1e742d10-8cdc-11e8-9efd-748017139e6c.png)
  
  If we convert this image to grayscale and use the ```inRange()``` function with parameters set to detect yellow and white lines, the output is not very clear in situations where shadows are present on the lane lines.
  ##### Color selection in grayscale image
  ![grayout](https://user-images.githubusercontent.com/8627486/43037471-47413482-8cdb-11e8-9757-5f26069dd0b3.png)
  
 I converted the original image to a HSV image and observed that the color selected output performs better than a color selected grayscale image. Although, in comparison with the above two formats, the color selected output of a HSL image was much better in detecting white and yellow lane lines in image and video data without getting affected by any shadow on the road. Hence, I perform the further image processing techniques on a HSL image.
 ##### Color selection in a HSL image
 ![hslout](https://user-images.githubusercontent.com/8627486/43037612-fc394eaa-8cdc-11e8-95c9-fc3ff8ec46a5.png)



### 2. Potential shortcomings of current pipeline


### 3. Possible improvements of current pipeline

