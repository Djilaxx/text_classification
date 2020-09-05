import numpy as np
import torch
from tqdm import tqdm

from utils.average_meter import AverageMeter
from utils.metrics import metrics_dict

import warnings
warnings.filterwarnings("ignore")

class Trainer:
    '''
    trn_function train the model for one epoch
    eval_function evaluate the current model on validation data and output current loss and other evaluation metric
    '''
    def __init__(self, model, optimizer, device, criterion, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.criterion = criterion

    def training_step(self, data_loader):
        self.model.train()
        losses = AverageMeter()

        tk0 = tqdm(data_loader, total=len(data_loader))

        for _, data in enumerate(tk0):
            images = data["images"]
            labels = data["labels"]

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.model.zero_grad()

            output = self.model(images)
            loss = self.criterion(output, labels)
            
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            losses.update(loss.item(), images.size(0))
            tk0.set_postfix(loss=losses.avg)

    def eval_step(self, data_loader, metric):
        self.model.eval()
        losses = AverageMeter()
        metrics_avg = AverageMeter()

        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader))
            for _, data in enumerate(tk0):
                images = data["images"]
                labels = data["labels"]

                images = images.to(self.device)
                labels = labels.to(self.device)

                output = self.model(images)
                loss = self.criterion(output, labels)

                metric_used = metrics_dict[metric]
                predictions = torch.softmax(output, dim=1)
                _, predictions = torch.max(predictions, dim=1)

                metric_value = metric_used(labels, predictions)

                losses.update(loss.item(), images.size(0))
                metrics_avg.update(metric_value.item(), images.size(0))

                tk0.set_postfix(loss=losses.avg)
        print(f"Validation Loss = {losses.avg}")
        return metrics_avg.avg
