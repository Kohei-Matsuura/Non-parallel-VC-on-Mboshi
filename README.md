# Non-parallel-VC-on-Mboshi
Following the steps explained below, you can reproduce my experiment.

## Abbreviations
VC: Voice Conversion  
ASR: Automatic Speech Recognition

## Purpose with one line
To improve low-resource ASR recognition accuracy with VC-based speaker adaptation

## Experimental setting
Here you can reproduce our experiment with [published Mboshi corpus](https://github.com/besacier/mboshi-french-parallel-corpus). This dataset contains 3 speakers, and we name them A, B, and C.

ASR model trained with A and B cannot recognize C's speeches well. Hence, it seems helpful to convert speeches of A and B into C-like speech and train the ASR model with C-like voices.

3 speakers and official train/dev sets divide the data into 6 parts as below.

|speaker|A|B|C|
|:---|:---|:---|:---|
|train|1|2|3|
|dev|4|5|6|

Voice conversion is trained with {1, 2} (source) and {3} (target).  
ASR is trained with {1, 2, converted 1, converted 2} and evaluated with {6}.

## Requirements
Python >= 3.6.0  
tqdm
PyTorch >= 1.0  
HCopy in [HTKTools](http://htk.eng.cam.ac.uk/) 
(If anything else, please give me an issue.)

## Introduction
1. Download or clone this repository

2. Download mboshi dataset from [here](https://github.com/besacier/mboshi-french-parallel-corpus). Then unzip it in your Downloads folder.

3. Run 'preprocess.recipe'
```
$ chmod 700 preprocess.recipe
$ ./preprocess.recipe
```  
and you can get   
- mboshi/train.script: ASR traning data
- mboshi/dev.lmfbs: ASR evaluation data, and used for VC training
- mboshi/dev.gt: ASR ground truth
- mboshi/train.lmfbs: used for Cycle-GAN training

4. ~~Download or clone VC modules from [here](https://github.com/Kohei-Matsuura/CycleGAN-VC2).~~  
(Edit on 8th May: Now it contains CycleGAN-VC2 directory.)  
Directory structure is as below.
```
Current Directory/  
    ├─ mboshi/   
    └─ CycleGAN-VC2/
```

5. Run 'cycle_gan.recipe'
```
$ chmod 700 cycle_gan.recipe
$ ./cycle_gan.recipe
```

Now you have ASR training script ('ASR.train.script'), which contains the converted features.

## Result
With [this ASR model](https://github.com/Kohei-Matsuura/LAS), PERs are as below.  
(The modeling unit is phone.)

|method|PER (%)|
|:---|:---|
|Baseline|44.0|
|VC-based Adaptation|25.9|
