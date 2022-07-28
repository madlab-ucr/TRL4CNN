'''
Created Date: Wednesday, July 27th 2022, 12:57:08 pm
Author: Rutuja Gurav (rutuja.gurav@email.ucr.edu)
Copyright (c) 2022 M.A.D. Lab @ UCR (https://madlab.cs.ucr.edu)

'''

## For experimental reproducibility fix a seed for all RNGs
seed_value = 1234
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
# import tensorflow as tf
# tf.random.set_seed(seed_value)
import torch
torch.manual_seed(seed_value)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

from datetime import datetime
import time, json, argparse, sys, logging
import torch.nn as nn
import torchmetrics
from dataloader.dataloader import get_dataloaders
from models.baseline import VGG16, BaselineCNN
from models.trl_models import TRL4CNN
from tqdm import tqdm
from utils import plot_training_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="baseline",
                    help='options: baseline, trl')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_DIR = '/home/rgura001/trl4cnn/results'
RUN_DIR = RESULTS_DIR+'/{}_run_{}'.format(args.model, datetime.now().strftime("%Y%m%d")) # "%Y%m%d-%H%M%S"
if not os.path.exists(RUN_DIR):
    os.makedirs(RUN_DIR)
    os.makedirs(RUN_DIR+'/plots')
    os.makedirs(RUN_DIR+'/eval_data')

## Setup your in-script loggers for debugging
logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)
LOG_FORMAT_STRING = ("%(levelname)s %(asctime)s - %(message)s")
LOG_FILENAME = RUN_DIR+"/logs"
log_format = logging.Formatter(LOG_FORMAT_STRING)
file_handler = logging.FileHandler(LOG_FILENAME, mode='a')
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.info("Getting data loaders...")
train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)

logger.info("Getting model...")
num_classes = 10
if args.model == 'baseline':
    model = BaselineCNN(num_classes)
elif args.model == 'trl':
    model = VGG16(num_classes)

### Move model to device
if torch.cuda.is_available():
    model.cuda()

logger.info("Saving model stats in info.txt...")
from torchinfo import summary
model_stats = summary(model, (1,3,32,32))
# print(model_stats)
logger.info("Model:\n {}".format(str(model_stats)))

logger.info("Training...")
def train(num_epochs, model, optimizer, scheduler, lossfn):
    dur = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    lrs = []
    for epoch in range(num_epochs):
        epoch_train_loss = []
        epoch_val_loss = []
        epoch_train_acc = []
        epoch_val_acc = []

        if epoch >=3:
            t0 = time.time()
        
        model.train()
        #Load in the data in batches using the train_loader object
        for i, (images, labels) in tqdm(enumerate(train_loader)):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            train_preds = model(images)
            train_loss = lossfn(train_preds, labels)
            train_acc = torchmetrics.functional.accuracy(train_preds, labels)

            epoch_train_loss.append(train_loss.item())
            epoch_train_acc.append(train_acc.item())

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        # Validation pass
        model.eval()
        for i, (images, labels) in tqdm(enumerate(val_loader)):  
            images = images.to(device)
            labels = labels.to(device)
            
            val_preds = model(images)
            val_loss = lossfn(val_preds, labels)
            val_acc = torchmetrics.functional.accuracy(val_preds, labels)
            
            epoch_val_loss.append(val_loss.item())
            epoch_val_acc.append(val_acc.item())
        
        scheduler.step()
        curr_lr = scheduler.get_last_lr()
        lrs.append(curr_lr)

        if epoch >=3:
            dur.append(time.time() - t0)
        train_losses.append(sum(epoch_train_loss)/len(epoch_train_loss))
        val_losses.append(sum(epoch_val_loss)/len(epoch_val_loss))
        train_accs.append(sum(epoch_train_acc)/len(epoch_train_acc))
        val_accs.append(sum(epoch_val_acc)/len(epoch_val_acc))
        print("Epoch {:05d} | Train Loss {:.4f} | Val Loss {:.4f} | Train Accuracy {:.4f} | Val Accuracy {:.4f} | Time(s) {:.4f}"
        .format(epoch, train_loss.item(), val_loss.item(), train_acc.item(), val_acc.item(), np.mean(dur)))
    return train_losses, val_losses, train_accs, val_accs, lrs

learning_rate = 1e-3
num_epochs = 20
lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

logger.info("\t Model: {}".format(args.model))
logger.info("\t learning_rate = {}".format(learning_rate))
logger.info("\t num_epochs = {}".format(num_epochs))
logger.info("\t optimizer = {}".format(optimizer.state_dict()))
logger.info("\t scheduler = {}".format(scheduler.state_dict()))

tic = datetime.now()
train_losses, val_losses, \
    train_accs, val_accs, lrs = train(num_epochs, model, optimizer, scheduler, lossfn)
logging.info("Done training. Elapsed time = {}".format(datetime.now()-tic))

logging.info("Plotting training mterics...")
history = {}
history['ce_loss'] = train_losses
history['val_ce_loss'] = val_losses
history['accuracy'] = train_accs
history['val_accuracy'] = val_accs
history['lr'] = lrs
with open(RUN_DIR+'/eval_data/history.json','w') as f:
    f.write(json.dumps(history))

## When running at-scale plot asychronously.
plot_training_metrics(history=history, title="", SAVE_PATH=RUN_DIR+'/plots/')

logger.info("Calculating metrics on testset...")
model.eval()
test_losses = []
test_accs = []
for i, (images, labels) in tqdm(enumerate(test_loader)):  
    images = images.to(device)
    labels = labels.to(device)
    
    test_preds = model(images)
    test_loss = lossfn(test_preds, labels)
    test_acc = torchmetrics.functional.accuracy(test_preds, labels)
    test_losses.append(test_loss.item())
    test_accs.append(test_acc.item())

cumul_test_accuracy = sum(test_accs)/len(test_accs)
logger.info("Test Accuracy = {}".format(cumul_test_accuracy))

## closing logger on exit
for handler in logger.handlers:
    handler.close()
    logger.removeHandler(handler)