# 2021 Korean Body Shape & Measurement Data AI Hackathon

## 1. Notes

    This is an replicated implementation of Sasang Physical Constitutions based on Upper Body.

## 2. Environment

    -

## 3. File Structure
main.py<br/>
┣ dataloaderV2.py<br/>
┣ augmentation.py<br/>
┣ train.py<br/>
┣ test.py<br/>
┣ loss.py<br/>
┣ score.py<br/>
┣ constitution_classifier.py<br/>
┗ utils<br/>
┃ ┗ files.py<br/>
┃ ┗ ftp.py<br/>
┗ config.py<br/>

## 4. Method
    1. train
        => Models **train** images and masks of the frontal upper body 
    2. test
        => Models **predict** images and masks of the frontal upper body 
    3. classify constitution
        => The post-processing predicts person's physical constitution using extracted width informations from mask.

### Segmentation Model 

    - DeepLabV3plus
    - EfficientNet-b3


    