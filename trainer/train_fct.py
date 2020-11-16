##################
# IMPORT MODULES #
##################
import numpy as np
import torch
from tqdm import tqdm
from utils.average_meter import AverageMeter
from utils.metrics import metrics_dict
import warnings
warnings.filterwarnings("ignore")
#################
# TRAINER CLASS #
#################
class Trainer:
    '''
    trn_function train the model for one epoch
    eval_function evaluate the current model on validation data and output current loss and other evaluation metric
    '''
    def __init__(self, model, optimizer, device, criterion):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion
    #################
    # TRAINING STEP #
    #################
    def training_step(self, data_loader):
        # LOSS AVERAGE
        losses = AverageMeter()
        # MODEL TO TRAIN MODE
        self.model.train()
        # TRAINING LOOP
        tk0 = tqdm(data_loader, total=len(data_loader))
        for _, data in enumerate(tk0):
            # LOADING IMAGES & LABELS
            ids = data["ids"]
            masks = data["masks"]
            labels = data["labels"]
            ids = ids.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            # RESET GRADIENTS
            self.model.zero_grad()
            # CALCULATE LOSS
            output = self.model(ids, masks)
            loss = self.criterion(output, labels)
            # CALCULATE GRADIENTS
            loss.backward()
            self.optimizer.step()
            # UPDATE LOSS
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg)
    ###################
    # VALIDATION STEP #
    ###################
    def eval_step(self, data_loader, metric, n_class):
        # LOSS & METRIC AVERAGE
        losses = AverageMeter()
        metrics_avg = AverageMeter()
        # MODEL TO EVAL MODE
        self.model.eval()
        # VALIDATION LOOP
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader))
            for _, data in enumerate(tk0):
                # LOADING IMAGES & LABELS
                ids = data["ids"]
                masks = data["masks"]
                labels = data["labels"]
                ids = ids.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                # CALCULATE LOSS & METRICS
                output = self.model(ids, masks)
                loss = self.criterion(output, labels)

                metric_used = metrics_dict[metric]
                predictions = torch.softmax(output, dim=1)
                _, predictions = torch.max(predictions, dim=1)

                metric_value = metric_used(labels, predictions, n_class)

                losses.update(loss.item(), ids.size(0))
                metrics_avg.update(metric_value.item(), ids.size(0))

                tk0.set_postfix(loss=losses.avg)
        print(f"Validation Loss = {losses.avg}")
        return loss, metrics_avg.avg
