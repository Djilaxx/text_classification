import os, inspect, importlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gc
import pandas as pd
import numpy as np
from pathlib import Path

#ML import
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import BertTokenizer
from sklearn import metrics
from sklearn import model_selection
#My own modules
from text_dataset import text_ds
from utils import early_stopping, folding, parser
from trainer.train_fct import Trainer
from .config import config

def run(folds=5, model="distilbert", metric="ACCURACY"):

    print(f"Training for {folds} with {model} model")

    #Creating the folds from the training data
    folding.create_folds(datapath=config.main.TRAIN_FILE,
                        output_path=config.main.FOLD_FILE,
                        nb_folds = folds,
                        method=config.main.FOLD_METHOD,
                        target=config.main.TARGET_VAR)

    #Load data
    df = pd.read_csv(config.main.FOLD_FILE)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    for fold in range(folds):
        print(f"Starting training for fold : {fold}")

        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        model = model(config.main.N_CLASS, config.main.DISTILBERT_PATH)
        model.to(config.main.DEVICE)

        #Create dataset and dataloader
        trn_text = df_train[config.main.IMAGE_ID].values.tolist()
        trn_text = [os.path.join(config.main.TRAIN_PATH, i) for i in trn_text]
        trn_labels = df_train[config.main.TARGET_VAR].values

        valid_text = df_valid[config.main.IMAGE_ID].values.tolist()
        valid_text = [os.path.join(config.main.TRAIN_PATH, i) for i in valid_text]
        valid_labels = df_valid[config.main.TARGET_VAR].values

        trn_ds = text_ds.TEXT_DS(
            text = trn_text,
            labels = trn_labels,
            tokenizer = tokenizer,
            max_len = config.main.MAX_lEN
        )

        train_loader = torch.utils.data.DataLoader(
            trn_ds, batch_size=config.hyper.TRAIN_BS, shuffle=True, num_workers=4
        )

        valid_ds = text_ds.TEXT_DS(
            text = valid_text,
            labels = valid_labels,
            tokenizer = tokenizer,
            max_len = config.main.MAX_LEN 
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_ds, batch_size=config.hyper.VALID_BS, shuffle=True, num_workers=2
        )

        #Set optimizer, scheduler, early stopping etc...
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.hyper.LR)
        scheduler = None
        es = early_stopping.EarlyStopping(patience=2, mode="max")
        trainer = Trainer(
            model, optimizer, config.main.DEVICE, criterion, scheduler=scheduler)

        #Starting training for nb_epoch
        for epoch in range(config.hyper.EPOCHS):
            print(f"Starting epoch number : {epoch}")
            #Training phase
            print("Training the model...")
            trainer.training_step(train_loader)
            #Evaluation phase
            print("Evaluating the model...")
            metric_value = trainer.eval_step(valid_loader, metric)
            #Metrics
            print(f"Validation {metric} = {metric_value}")

            Path(os.path.join(config.main.PROJECT_PATH, "model_output/")).mkdir(parents=True, exist_ok=True)
            es(metric_value, model,
               model_path=os.path.join(config.main.PROJECT_PATH, "model_output/", f"model_{fold}.bin"))
            if es.early_stop:
                print("Early Stopping")
                break
            gc.collect()

#Create the parser
args = parser.create_parser()

if __name__ == "__main__":
    print("Training start...")

    run(
        folds=args.folds,
        model=args.model,
        metric=args.metric
    )



