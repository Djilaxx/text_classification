# **Text classification**

This repository contains text classification projects i've worked on, either for fun/education or competition on Kaggle. \
Each project have it's own readme containing information about the specific problematics of each. 

I train the models locally on my pc using a Nvidia 1080 GPU. 
## **Data**

The data is not in the repository directly if you want to launch a model on one the projects in here you must download the data and change the config file in the task folder to be adequate. \
Links to the datasets are in the tasks Readme.
## **Task**
---
The task folder contains the specific code about each project :
 * config.py file containing  most of the hyperparameters of the model.
 * run.py file that start the training cycle. 
 * inference.py that allow to use a trained model for inference on test data

To start training a model on any task use this command in terminal :
```
python -m task.tweet_disaster.run
```
You can replace the **tweet_disaster** with any folder in task. (or a task you want to add)
Default parameters train for **5** fold using a **distilbert** model and return validation **accuracy** after each epoch. 
You can change these parameters as such :
```
python -m task.tweet_disaster.run --folds=3 --model=bert --metric=F1_SCORE
```
PS : ATM only distilbert is available as a model, i plan to add more transformers and also RNNs.

The parameters can take different values :
* **folds** : this parameter determine the number of folds to create into the dataset. If you choose 5 for example, the dataset will be divided in 5, train a model on 4 folds and validate on the last (folds 0, 1, 2, and 3 for training and 4 for validation. Then, it'll train on folds 0, 1, 2, 4 and validate on 3 etc...).
* **model** : You can choose any model that is in the models/ folder, name must be typed in MAJ like in the example above.
* **metric** : The training procedure return loss throughout training and validation, and accuracy by default, but you can choose to return another validation metric if needed (the metrics you can choose from are in **utils/metrics.py**)

## **Text Dataset**
---
The text dataset folder contain a dataset class that loads text and corresponding labels, and transforms the text in the correct format for the chosen model (atm it returns ids, masks and labels as expected to train a distilbert model)

## **Models**
---
The models folder contain the different models that can be used on any task. \
The models i use follow a similar structure, they are pretrained models to which i add a final linear layer for classification.

The model output is directly coming from the final layer. If you want to obtain probabilities you should pass the output through a softmax/sigmoid function.
## **Trainer** 
---
The trainer is a class that is used to train the model for one epoch and validate afterward. It contains a training function and an evaluation function.

## **To do** 
---
* Being able to switch between transformers and RNNs seemlessly.
* Add an inference file for each task
* Add scheduler
* Add more models
* Add TPU support for the dataset and training. 
* Add metrics
* Debug the AUROC metric (not usable atm)
* Add logger to have information about each training run you perform