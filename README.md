# SIIM-ACR Pneumothorax Segmentation (Kaggle)
**Identify Pneumothorax disease in chest x-rays**

## My Silver Medal Solution (Ranked: 47/1475)
This competition is being hosted by the Society for Imaging Informatics in Medicine (SIIM) in coordination with the Annual Conference on Machine Intelligence in Medical Imaging (C-MIMI)
The top winning teams in the competition will be awarded with monetary prizes up to **$30,000**,
and each winning team will be invited and strongly encouraged to attend the conference with waived registration fees,
contingent on review of solution and fulfillment of winners' obligations. 
For this challenge, participants will develop a model to classify (and if present, segment)
pneumothorax from a set of chest radiographic images. 
If successful, you could aid in the early recognition of pneumothoraces and save lives.

In this solution, we split the task into 2 stages. 
We first create a classication model to identify if a patient is presence with pneumothorax and if it does appear,
we will carry out with the segmentation prediction. 
For convenience, we predict the segmenation map for all cases in our implementation and we will correct it with the classification result before saving as our final submission.

## Classification Model (Part 1)
In this section, we built 3 different classifiers to detect our patients with pneumothorax and 
the models are built on the backbone of EfficientNet B4 and Xception architecture, fine-tuned with the loaded pre-trained weights.
Then, we add some tricks to our data augmentation by adding a bounding box crop that dissect the region of chest to remove any redundant pixels. 
This will allow the machine to focus and learn better on the identifying features of the chest. However, to get the bounding boxes, 
we built a Faster R-CNN model with 1000 hand labelled targets provided by Dr Konya. 
The original bounding box labels and the one from our model can be accessed in the [boundingbox_csv](https://github.com/DW-Hwang/SIIMS_ACR_Pneumothorax-Kaggle/tree/master/boundingbox_csv) folder.
Additionally, the results from our model is shown below.

<img src="https://github.com/DW-Hwang/SIIMS_ACR_Pneumothorax-Kaggle/blob/master/screenshots/image1.png" width= "768" height="640"/>

Then, we ensembled our model by stacking the three models with a fully connected layers from the bottleneck output of the three models,
where we froze up the front part and train only on the fully connected layers. At last, we use it to predict our classification results.


## Segmentation Model (Part 2)
