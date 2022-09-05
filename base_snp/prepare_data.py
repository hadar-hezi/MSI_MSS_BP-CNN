# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:17:36 2020
@author: HADAR1
"""
import torch
import numpy as np
import os
import random
import torchvision
from torchvision import datasets, models, transforms
from sklearn.model_selection import StratifiedKFold
import pandas as pd

from new_dataset import CustomImageFolder

# for reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
       
class Prepare():
    def __init__(self,hp):
      self.dataset_sizes = None
      self.class_names = None
      self.datasets,self.dataloaders = {},{}
      self.train_batch_size = hp['batch_size']
      self.valid_batch_size = hp['batch_size']
      self.num_workers = hp['num_workers']
      self.num_classes = hp['num_classes']
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.data_dir = hp['data_dir']
      self.train_patients = get_patients_list(f"{hp['root_dir']}train_patients.csv")
      self.test_patients = get_patients_list(f"{hp['root_dir']}test_patients.csv")
      # separate patients to train and valid sets
      # validation size is 1/n_splits
      self.skf = StratifiedKFold(n_splits=hp['n_folds'],random_state=hp['newseed'],shuffle=True)
      snp_labels = get_snp_label(hp,self.train_patients)
      self.train_patients = np.array(self.train_patients)
      snp_labels = np.array(snp_labels)
      # labels contain negative labels
      pos_ind = np.nonzero(snp_labels >= 0)
      snp_labels = snp_labels[pos_ind]
      self.train_patients = self.train_patients[pos_ind]
      # iterator for folds
      self.skf_it = self.skf.split(self.train_patients,snp_labels)
      
    def prepare_data(self,hp):
    # Data augmentation and normalization for training
    # Just normalization for validation
        if hp['model_type'] == 'inception':
            size = 299
            resize = 350
        else:
            size = 224
            resize = 256                                   
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size),
                # transforms.RandomAffine(degrees=60),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }
       

        self.train_names, self.valid_names = patients_partition(hp,self.skf_it ,self.train_patients)
        print("train patients size: ", len(self.train_names),"valid patients size: ",  len(self.valid_names))
        # create datasets
        test_dataset= CustomImageFolder(root=os.path.join(self.data_dir,'test'), transform= data_transforms['val'],hp=hp)
        train_data = CustomImageFolder(root=os.path.join(self.data_dir,'train'), transform= data_transforms['train'],patients_list=self.train_names,hp=hp)
        valid_data = CustomImageFolder(os.path.join(self.data_dir,'train'),  data_transforms['val'],patients_list=self.valid_names,hp=hp)
        #weighted sampler
        self.train_weights,self.train_weights_perclass = self.make_weights_for_balanced_classes(train_data.imgs, self.num_classes)
        self.valid_weights, self.valid_weights_perclass = self.make_weights_for_balanced_classes(valid_data.imgs, self.num_classes)                                                                                                                                     
        # for reproducibility
        self.g = torch.Generator()
        self.g.manual_seed(0)
  
        self.class_names = train_data.classes
        self.datasets = {'train': train_data, 'valid': valid_data,'test': test_dataset}
        self.dataset_sizes = {'train': len(train_data), 'valid': len(valid_data),'test': len(test_dataset)}
        print(self.dataset_sizes)

    def create_train_validation_loaders(self):
        """
        Splits a dataset into a train and validation set, returning a
        """
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(self.train_weights, self.dataset_sizes['train'] )
        valid_sampler = torch.utils.data.sampler.WeightedRandomSampler(self.valid_weights, self.dataset_sizes['valid'])
        dl_train = torch.utils.data.DataLoader(self.datasets['train'],
                                               batch_size=self.train_batch_size,
                                                sampler = train_sampler,
                                                #shuffle=True,
                                               num_workers = self.num_workers,
                                               worker_init_fn=seed_worker,
                                               generator=self.g)
        dl_valid = torch.utils.data.DataLoader(self.datasets['valid'],
                                               batch_size = self.valid_batch_size,
                                               #shuffle=True,
                                               sampler=valid_sampler,
                                               num_workers = self.num_workers,
                                               worker_init_fn=seed_worker,
                                               generator=self.g)
        
        self.dataloaders['train']= dl_train
        self.dataloaders['valid']= dl_valid
        
    def create_test_loader(self):
        dl_test = torch.utils.data.DataLoader(self.datasets['test'],
                                                batch_size = self.valid_batch_size,
                                                shuffle=True,
                                                # sampler=test_sampler,
                                                num_workers = self.num_workers,
                                                worker_init_fn=seed_worker,
                                                generator=self.g)
        self.dataloaders['test']= dl_test
        # self.loaded_ind['test'] = test_sampler.get_indices()
#TODO move out from class
    def make_weights_for_balanced_classes(self,samples, nclasses):                        
      """
        
        Parameters
        ----------
        samples (tuple): (str,int,int) contains the dataset instances (path,label,sub_label).
        nclasses (int):  number of classes

        Returns
        -------
        weight (list) of size n_samples: weight for each sample
        weight_per_class (list): weight is n_samples/n_class_samples
      
        """
      count = [0] * nclasses                                                 
      for item in samples: 
          # index 1 is the label
          count[item[1]] += 1                                                       
      weight_per_class = [0.] * nclasses   
      # N total number of samples
      N = float(sum(count))  
      samples_count = nclasses*min(count)
      for i in range(nclasses): 
          if count[i] > 0:
              weight_per_class[i] = N/float(count[i])   
          print(f"weight_per_class {i}: {weight_per_class[i]}")                              
      weight = [0] * len(samples)                
       # the sample weight is the class weight                             
      for idx, val in enumerate(samples):  
          if val[1] > -1:
              weight[idx] = weight_per_class[val[1]]   
      weight = torch.DoubleTensor(weight)   
      
      return weight,weight_per_class   
  
     # loads model checkpoint 
    def load_model(self,model_file,model):
       
        print(f"*** Loading checkpoint file {model_file}")
        saved_state = torch.load(model_file, map_location=self.device)
        model.load_state_dict(saved_state["model_state"])
        
        return model,saved_state
    
   # define unfixed layers of the model    
    def train_params(self,hp,model):
    # for efficient net
        if hp['model_type'] == 'efficient':
            for param in model.parameters():
                param.requires_grad = False
            for param in model._blocks[48:].parameters():
                param.requires_grad = True
    # for inception
        if hp['model_type'] == 'inception':
            for param in model.Mixed_7a.parameters():
                param.requires_grad = True
            for param in model.Mixed_7b.parameters():
                param.requires_grad = True
            for param in model.Mixed_7c.parameters():
                param.requires_grad = True
            
        return model

# loads patient names     
def get_patients_list(path):
    """
    path (str): path to a csv file with train or test patient names
    
    Returns:
       patient_names (list): sorted list of the patient names 
        
    """                                
    df = pd.read_csv(path) 
    names = list(df['name'])
                                   
    # unique names list
    patient_names = list(set(names)) 
    patient_names.sort()
    return patient_names

def patients_partition(hp,skf_it,patients_list):
    """
    

    Parameters
    ----------
    hp : dict
        hyper parameters dictionary.
    skf_it : iterator
        iterates the generated folds.
    patients_list : list
        list of all train patients.

    Returns
    -------
    train_names : list
        train names in the current fold.
    valid_names : list
         validation names in the current fold..

    """
    # get the next fold
    train_index, valid_index = next(skf_it)
                                                                                             
    train_names, valid_names =patients_list[train_index],patients_list[valid_index]
    return train_names, valid_names
     
def get_snp_label(hp,patients):   
    """
    

    Parameters
    ----------
    hp : dict
         hyper parameters dictionary.
    patients : list
        full train set patient names.

    Returns
    -------
    return_label : list
        sub labels defined by snp.

    """
    return_label = []
    # Data for sub-label
    df = pd.read_csv(f"{hp['root_dir']}data_table_Colorectal_Adenocarcinoma.csv")
    patients_sub_mss = list(df['patient'])
    label = list(df['snp label'])
    for patient in patients:
        for i,sub_patient in enumerate(patients_sub_mss):
            if sub_patient == patient:
                return_label.append(label[i])
    return return_label 
    
    