#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import numpy as np
import cv2
from tqdm import tqdm_notebook as tqdm

#sys.path.append("/content/drive/My Drive/CESI/Projets A5/Data Science/Projet DataScience/Pipeline")
from Pipeline.Degradation import UglyImage

class Pipeline():
    def __init__(self, dataset_name = 'train2014', img_size=(512, 512)): # no *args or **kargs
        self.dataset_name = dataset_name
        self.img_size = img_size

    def create_tree_directories(self):
        os.system('mkdir data/Train')
        os.system('mkdir Train/degraded')
        os.system('mkdir Test')

    def download_data(self):
        start_time = time.time()
        url = "images.cocodataset.org/zips/"+self.dataset_name+".zip"  # URL du dataset
        os.system('wget %s' %url) # Téléchargement de l'archive
        os.system('unzip %s' %self.dataset_name+'.zip')  # Dézippage de l'archive
        os.system('mv %s' %self.dataset_name+' Train/clean')
        os.remove('/content/'+self.dataset_name+'.zip')  # Suppression de l'archive
        data_path = os.getcwd()+"/Train/clean" # Récupération de l'emplacement des données récoltées
        print("Temps d execution : %s secondes ---" % (time.time() - start_time)) # Affichage du temps d'exécution 
        return data_path

    def download_git_data(self):
        url = "https://github.com/Pielgrin/dataset_clean_degraded.git"
        os.system('git clone %s' %url)
        os.system('mv dataset_clean_degraded Val')
        os.system('mv Val/test_degraded Test')
