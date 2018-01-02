# pix2pix_tensorflow

Just for PoC!
- Need to tune the ratio of 'L2' and 'GAN' losses of Generator!
- Need to apply data augmentation..  ( but, just enough to prove the concept! This is all I want )

Used 'facedes' dataset from 'https://github.com/affinelayer/pix2pix-tensorflow', great thanks!
The above site provides with good and exact example, 
just simple is mine! 

Do : 
1. download dataset 
  -  python download-dataset.py facades
2. adjust datase folder
  -  In pix2pix.py, at line 20~22, please set the proper dataset path as yours.
3. run
    python pix2pix.py   



