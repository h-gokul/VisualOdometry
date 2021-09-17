# Visual Odometry:
We use  filtered SIFT keypoint correspondences from RANSAC utilising the epipolar constraints to obtain the essential matrix, which is used to compute the camera poses. Cheirality Check is performed to ensure that the physically correct pose is selected out of mathematically possible 4 poses. 

download Oxford Robot Car dataset from [here](https://drive.google.com/drive/folders/1hAds4iwjSulc-3T88m9UDRsc6tBFih8a) and extract in ./Data folder

Run 
```
python3 VisualOdometry.py
```
