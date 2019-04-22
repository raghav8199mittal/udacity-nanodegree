import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from PIL import Image
import json

def main():
    input=get_args()
    path_img=input.image_path
    savep=input.checkpoint
    n=input.top_k
    name_cat=input.category_names
    gpu=input.gpu
    #getting names
    with open(cat_names,'r') as i:
        cat_to_name=json.load(i)
        
    model=load(savep)
    img=Image.open(path_img)
    image=process_image(img)
    probs,classes=predict(path_img,model,n)
    check(image,path_img,model)
    
#define fun get args
def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("image_path",type=str,help="path of the image")
    parser.add_argument("checkpoint", type=str,help="model is contained")
    parser.add_argument("--top_k",type=int,default=10,help="number")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="index to name")
    parser.add_argument("--gpu", type=bool, default=True,help="Train model with GPU")
    return parser.parse_args()

#Neuralnet
class NeuralNetwork(nn.Module):
    def __init__(self,input_size,output_size,hidden_layers):
        super().__init__()
        self.hidden_layers=nn.ModuleList([nn.Linear(input_size,hidden_layers[0])])
        
        a=0
        b=len(hidden_layers)-1
        while a != b:
            l=[hidden_layers[a],hidden_layers[a+1]]
            self.hidden_layers.append(nn.Linear(l[0],l[1]))
            a+=1
        
        for each in hidden_layers:
            print(each)
        self.output= nn.Linear(hidden_layers[b],output_size)
        
    def forward(self, tensor):
        for linear in self.hidden_layers:
            tensor= F.relu(linear(tensor))
        tensor= self.output(tensor) 
        return F.log_softmax(tensor, dim=1)
    
#load
def load(im):
    checkpoint=torch.load(im)
    model=getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    for ar in model.parameters():
        ar.requires_grad=False
    classifier=NeuralNetwork(checkpoint['input_size'],checkpoint['output_size'],checkpoint['hidden_layers'],checkpoint['drop'])
    model.classifier= classifier
    model.classifier.load_state_dict(checkpoint['state_dict'])
    model.classifier.optimizer=checkpoint['optimizer']
    model.classifier.epochs=checkpoint['epochs']
    model.classifier.learning_rate=checkpoint['learning_rate']
    return model
def process_image(image):
    wid,hei=image.size
    if wid==hei:
        size=256,256
    elif wid>hei:
        temp=wid/hei
        size=256*temp,256
    elif hei>wid:
        temp=hei/wid
        size=256,256*temp
    image.thumbnail(size, Image.ANTIALIAS)
    image=image.crop((size[0]//2 - 112, size[1]//2-112, size[0]+112,size[1] - 112))
    #color
    img_array=np.array([0.485,0.456,0.406])
    std=np.array([0.229,0.224,0.225])
    image=(np_image - mean)/std
    img=image.transpose((2,0,1))
    return img
def predict(path_im,model,num):
    im=Image.open(path_im)
    image=process_image(im)
    output=model(image)
    probs,indices=output.topk(topk)
    index_to_class={val: key for key,val in cat_to_name.items()}
    max_class=[index_to_class[each] for each in indices]
    return probs,max_class
def check(image,path_im,model):
    probs,classes=predict(path_im,model)
    sb.countplot(y=classes, x= probs,color='blue',ecolor='black',align='center')
    plt.show()
    ax.imsow(image)
if __name__ =="main":
    main()