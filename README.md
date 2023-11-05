## Index
* 1.) Overview and Purpose
* 2.) ฺBrief Summary
* 3.) Target and Matrix measurement
* 4.) Dataset
* 5.) Experiment and Result
* 6.) Domain Knowledge
<br/>

>> 1.) Overview and Purpose --------------------------------------------------------------------------------------------------------------
<br/>
<img src="https://github.com/Bawontat/pneumonia_X_ray_classify/assets/39905133/ebf9cfa4-78b7-4dbb-b68c-1cfe74dc0e9b" alt="image" width="193" height="auto">
<img src="https://github.com/Bawontat/pneumonia_X_ray_classify/assets/39905133/91cf7d90-2889-46a5-aea7-32c1ca2599be" alt="image" width="200" height="auto"> <br/>

I want to apply Image processing and machine learning for classify Pneumonia ans normal by X-ray film dataset.<br/>
I hope this model may be help doctor to screening the patient for split normal people and people likely to sick
<br/>
<br/>

>> 2.) Brief Summary --------------------------------------------------------------------------------------------------------------------

**Test dataset**       : 624 
                         (234 : Normal  , 390 : Pneumonia)

**Backbone model**     : EfficientNet V2_imagenet21k_xl

**Target**             : Every patient should be detect (Recall rate close to 100%)

|         |EfficientNet V2|
|:-------:|:-------:|
|Recall   |99.74 %|
|Precision|84.5 %|
<br/>

**Weight file**             :  [Download Weight](https://drive.google.com/drive/folders/1UuDs9aNEx7jdY5F6KAsU4Cv4npO8FDy7?usp=sharing)
<br/>
**Library (py3.10.10)**             : tensorflow 2.12.0 , pandas 2.0.1 , numpy 1.23.5
<br/>
<br/>


>> 3.) Target and Matrix measurement --------------------------------------------------------------------------------------------------------------

**" In real situations misdiagnosis of illness It is better than being misdiagnosed as not being sick. "**

Impact misdiagnosis of illness        : Waste time and money for continue health check with other tool.<br/>
Impact misdiagnosed as not being sick : May be die
<br/>
<br/>
So,  The target of confusion matrix is
<br/>
<br/>

<img src="https://github.com/Bawontat/pneumonia_X_ray_Classification/assets/39905133/247c9320-c747-495b-a4d5-59b6e76812d6" alt="image" width="500" height="auto">
<br/>

**so, Recall rate close to 100%**

>> 4.) Dataset -------------------------------------------------------------------------------------------------------

[Download Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

|         |Pneumonia [Bacteria , Virus]|Normal|
|:-------:|:-------:|:-------:|
|Train set   |[2530 , 1345]|1341|
|Validation set   |[8 , 0]|8|
|Test set   |[242 , 148]|234| </br>
* The model doesn't sperate detection between bacteria and virus


>> 5.) Experiment and Result -----------------------------------------------------------------------------------------------------

**5.1) Rapid experiment for estimate feasibility of detection by compare 3 backbone**  <br/>
*Use the same clasifier for 3 backbone

<img src="https://github.com/Bawontat/pneumonia_X_ray_Classification/assets/39905133/20006215-0a9b-4c15-bb33-ba9aa760e4de" alt="image" width="600" height="auto">
<br/>

EfficientNet V2     (Image size : 512x512 )   [Link](https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2) <br/>
Inception    V3     (Image size : 299x299 )   [Link](https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5) <br/>
Resnet       V2-152 (Image size : 224x224 )   [Link](https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5) <br/>

**Result**
<img src="https://github.com/Bawontat/pneumonia_X_ray_Classification/assets/39905133/64d156d9-e22e-408d-8cf3-2fed3e7b5b6e" alt="image" width="1000" height="auto"> <br/>

**Choose EfficientNetV2 and ResnetV2-152** <br/>

**5.2) Inference 2 models with test set**  <br/>
**Result**
<img src="https://github.com/Bawontat/pneumonia_X_ray_Classification/assets/39905133/99408964-dae8-4cc8-b337-05f8776dc017" alt="image" width="1200" height="auto"> <br/>

**5.3) Optimize model by cutting some layer of classifier**  <br/>
<img src="https://github.com/Bawontat/pneumonia_X_ray_Classification/assets/39905133/2df0167b-ac5a-46f0-8db5-b0e1c5b9ffad" alt="image" width="400" height="auto"> <br/>
**Result** <br/>
<img src="https://github.com/Bawontat/pneumonia_X_ray_Classification/assets/39905133/421924ce-9416-4f6b-aa28-30ed5dff26a0" alt="image" width="500" height="auto"> <br/>

>> 6.) Domain Knowledge ----------------------------------------------------------------------------------------------------------
<br/>
I attach image file about pneumonia knowledge, method of reading X-ray etc.<br/>
<br/>

[Gdrive link](https://drive.google.com/drive/folders/1ZU_tdZxlB5qRUDtN5ffxwHZM69GVwm-G?usp=sharing)

