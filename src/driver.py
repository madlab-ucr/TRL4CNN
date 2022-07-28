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
import time, json, argparse
import torch.nn as nn
import torchmetrics
from dataloader.dataloader import get_dataloaders
from models.baseline import BaselineCNN
from models.trl_models import TRL4CNN
from tqdm import tqdm
from utils import plot_training_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="baseline",
                    help='options: baseline, trl')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_DIR = '/home/rgura001/trl4cnn/results'
MAIN_RUN_DIR = RESULTS_DIR+'/{}_run_{}'.format(args.model, datetime.now().strftime("%Y%m%d")) # "%Y%m%d-%H%M%S"
if not os.path.exists(MAIN_RUN_DIR):
    os.makedirs(MAIN_RUN_DIR)
    os.makedirs(MAIN_RUN_DIR+'/plots')
    os.makedirs(MAIN_RUN_DIR+'/eval_data')

## Get dataloaders
train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)

## Get model
num_classes = 10
if args.model == 'baseline':
    model = BaselineCNN(num_classes)
elif args.model == 'trl':
    model = TRL4CNN(num_classes)

### Move model to device
if torch.cuda.is_available():
    model.cuda()

from torchinfo import summary
model_stats = summary(model, (1,3,32,32))
print(model_stats)

with open(MAIN_RUN_DIR+'/info.txt','w') as f:
    f.write("Model:\n {}".format(str(model_stats)))

## Train loop
def train(num_epochs, model, optimizer, lossfn):
    dur = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
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

        if epoch >=3:
            dur.append(time.time() - t0)
        train_losses.append(sum(epoch_train_loss)/len(epoch_train_loss))
        val_losses.append(sum(epoch_val_loss)/len(epoch_val_loss))
        train_accs.append(sum(epoch_train_acc)/len(epoch_train_acc))
        val_accs.append(sum(epoch_val_acc)/len(epoch_val_acc))
        print("Epoch {:05d} | Train Loss {:.4f} | Val Loss {:.4f} | Train Accuracy {:.4f} | Val Accuracy {:.4f} | Time(s) {:.4f}"
        .format(epoch, train_loss.item(), val_loss.item(), train_acc.item(), val_acc.item(), np.mean(dur)))
    return train_losses, val_losses, train_accs, val_accs

learning_rate = 1e-3
num_epochs = 5

lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
train_losses, val_losses, \
    train_accs, val_accs = train(num_epochs, model, optimizer, lossfn)

history = {}
history['ce_loss'] = train_losses
history['val_ce_loss'] = val_losses
history['accuracy'] = train_accs
history['val_accuracy'] = val_accs
with open(MAIN_RUN_DIR+'/eval_data/history.json','w') as f:
    f.write(json.dumps(history))

## When running at-scale plot asychronously.
plot_training_metrics(history=history, title="", SAVE_PATH=MAIN_RUN_DIR+'/plots/')

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
print("Test Accuracy = {}".format(cumul_test_accuracy))
with open(MAIN_RUN_DIR+'/info.txt','a') as f:
    f.write("\n\n\nAccuracy on testset = {}".format(cumul_test_accuracy))
