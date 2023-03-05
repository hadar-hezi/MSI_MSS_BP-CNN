#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 11:03:05 2020

@author: hadar.hezi@bm.technion.ac.il

"""

import torch
import numpy as np
from collections import namedtuple
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
import time
import copy

import log_file
from log_file import summary
from compute_roc import *
from plot import plot_loss_acc
from get_auc import *

    
def train_model(model, criterion, optimizer,prepare, hp,saved_state=None):
    since = time.time()
    best_auc = 0.0
    lr = hp['lr']
    if saved_state is not None:
        lr = saved_state.get('lr',lr)
        best_acc = saved_state.get("best_acc", best_acc)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    Result = namedtuple("Result", "f1 loss")
    train_res = Result(f1=[], loss=[])
    valid_res = Result(f1=[], loss=[])
    train_data_path = f"{hp['root_dir']}{hp['train_res_file']}_{hp['curr_fold']}.csv"
    model_checkpoint =  f"{hp['root_dir']}{hp['checkpoint_save']}_{hp['curr_fold']}.pt"
    num_epochs = hp['num_epochs']
    for epoch in range(num_epochs):    
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            epoch_preds,epoch_paths,epoch_labels,epoch_sub_labels, epoch_pos_probs = [],[],[],[],[]
            if phase == 'train':
                model.train()  # Set model to training mode
                    
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels,sub_labels, paths in prepare.dataloaders[phase]:
                paths = np.array(paths)
                inputs = inputs.to(prepare.device)
                labels=labels.to(prepare.device)
                sub_labels=sub_labels.to(prepare.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    newOutputs = outputs
                    if hp['model_type'] == 'inception':                                     
                        if phase=='train':
                            newOutputs = outputs[0]
                    prob_y = F.softmax(newOutputs, dim=-1)
                  
                    _, preds = torch.max(prob_y, -1)
                    # send to creiterion labels OR sub_labels
                    loss = criterion(newOutputs, sub_labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                              
                    # MSI probability is the max value between MSI1 and MSI2
                    prob_y_msi = torch.where(prob_y[:,0] > prob_y[:,2],prob_y[:,0],prob_y[:,2])
                    # save batch results
                    epoch_preds.append(preds)
                    epoch_labels.append(labels)
                    epoch_sub_labels.append(sub_labels)
                    epoch_pos_probs.append(prob_y_msi)
                    epoch_paths.append(paths)

                    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                # calculate accuracy with labels OR sub_labels
                running_corrects += torch.sum(preds ==sub_labels.data)
            # save epoch results            
            epoch_preds_tensor = torch.cat(epoch_preds)
            epoch_labels_tensor = torch.cat(epoch_labels)
            epoch_sub_labels_tensor = torch.cat(epoch_sub_labels)
            epoch_pos_probs_tensor = torch.cat(epoch_pos_probs)
            epoch_paths_tensor = np.hstack(epoch_paths)
            epoch_preds_tensor=epoch_preds_tensor.cpu().detach()
            epoch_sub_labels_tensor = epoch_sub_labels_tensor.cpu()
    
                        
            epoch_loss = running_loss / prepare.dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / len(epoch_preds_tensor)) *100
            f1 = f1_score(epoch_sub_labels_tensor,epoch_preds_tensor,average='weighted')
            if phase == 'train':
                train_res.f1.append(f1)
                train_res.loss.append(epoch_loss)
                train_summary = summary(epoch_preds_tensor,epoch_labels_tensor,epoch_sub_labels_tensor,epoch_pos_probs_tensor,epoch_paths_tensor)
            else:
                valid_res.f1.append(f1)
                valid_res.loss.append(epoch_loss)
                valid_summary = summary(epoch_preds_tensor,epoch_labels_tensor,epoch_sub_labels_tensor,epoch_pos_probs_tensor,epoch_paths_tensor)
                # calculate AUC patient level
                valid_auc = get_auc(valid_summary,prepare,hp=hp,mode='valid')
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print(f"{phase} f1: {f1}")
            # deep copy the model
            if phase == 'valid' and valid_auc > best_auc:
                best_auc = valid_auc
                print("**saving model")
                model.train()
                saved_state = dict(
                        lr = lr,
                        best_auc=best_auc,
                        ewi=epochs_without_improvement,
                        model_state=model.state_dict(),
                    )
                torch.save(saved_state, resnext_checkpoint)
                # reduce step size
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr*0.97
        
    log_file.save_results(train_data_path, train_summary)
    log_file.save_results(f"{hp['root_dir']}{hp['valid_res_file']}_{hp['curr_fold']}.csv", valid_summary)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val AUC: {:4f}'.format(best_auc))
    plot_loss_acc(train_res,valid_res,hp)

  
        
    del loss
    del prepare.dataloaders['train']
    del best_model_wts
    return model,saved_state



