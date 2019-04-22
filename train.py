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

resnet18 = models.resnet18(pretrained=True)
vgg16=models.vgg16(pretrained=True)
alexnet=models.alexnet(pretrained=True)

def main():
    input=get_args()
    data_dir=input.data_directory
    save_to=input.save_dir
    pretrained_model=input.arch
    learning_rate=input.learning_rate
    ep=input.epochs
    hidden_layers=input.hidden_units
    output_size=input.output
    gpu=input.gpu
    drop=0.2
    
    train_dir=data_dir+'/train'
    valid_dir=data_dir+'/valid'
    test_dir=data_dir+'/test'
    
    trainloader,validloader,testloader=process(train_dir,valid_dir,test_dir)
    
    model_dict={"vgg":vgg16,"resnet":resnet18,"alexnet":alexnet}
    input_size_dict={"vgg":25088,"resnet":512,"alexnet":9216}
    model = model_dict[pretrained_model]
    input_size=input_size_dict[pretrained_model]
    #model parameters
    for ar in model.parameters():
        ar.requires_grad=False
        
        #criterion
    classifier=NeuralNetwork(input_size,output_size,hidden_layers,drop)
    model.classifier=classifier
    criterion =nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print("Training Loss:")
    train(model,trainloader,validloader,criterion,optimizer,ep,gpu)
    #accur
    test_loss=accuracy(testloader,model,criterion,gpu)
    print("Accuracy (using test data):")
    print(test_loss)
    
    #savepoint
    checkpoint={"input_size":input_size,"output_size":output_size,
                "hidden_layers":hidden_layers,"drop":drop,"epochs":ep,"learning_rate":learning_rate,"arch":pretrained_model,
                "optimizer":optimizer,"state_dict":model.classifier.state_dict()}
    torch.save(checkpoint,save_to)
    
def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("data_directory", type=str,help="directory containing data")
    parser.add_argument("--save_dir",type =str,default="checkpoint_2.pth",help="saving trained images")
    parser.add_argument("--arch" ,type=str,default="vgg",help="pretrained models")
    parser.add_argument("--learning_rate",type=float,default=0.001,help="error with leaning")
    parser.add_argument("--epochs", type=int,default=1,help="numers of times to train")
    parser.add_argument("--hidden_units",type=list,default=[700,300],help="hidden layers")
    parser.add_argument("--gpu", type=bool, default= True, help="use gpu to train ,cup:false")
    parser.add_argument("--output", type=int ,default=102,help="output size")
    return parser.parse_args()

def process(train_dir,valid_dir,test_dir):
    train_transforms=transforms.Compose([transforms.RandomRotation(25),transforms.RandomResizeCrop(224),transforms.RandomHorizontalFlip(),trasforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    test_transforms=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    validation_transforms=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    #dataset
    train_data=datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data=datasets.ImageFolder(test_dir,transform=test_transforms)
    valid_data= datasets.ImageFolder(valid_dir, transform=validation_transforms)
    
    #load data
    trainloader=torch.utils.data.DataLoader(train_data,batch_size=64, shuffle=True)
    testloader=torch.utils.data.DataLoader(test_data,batch_size=64, shuffle=True)
    validloader=torch.utils.data.DataLoader(valid_data,batch_size=64, shuffle=True)
    return trainloader,validloader,testloader

class NeuralNetwork(nn.Module):
    def __init__(self,input_size,output_size,hidden_layers,drop):
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
        self.dropout=nn.Dropout(p=drop)
    def forward(self, tensor):
        for linear in self.hidden_layers:
            tensor= F.relu(linear(tensor))
            tensor=self.dropout(tensor)
        tensor= self.output(tensor) 
        
        return F.log_softmax(tensor, dim=1)


    
def train(model,trainloader,validloader,criterion,optimizer,epochs,gpu):
    le=len(validloader)
    print_every=100
    steps=0
    model.to('cuda')
    e=0
    while e < epochs:
        run_loss=0
        losval=0
        for i,l in iter(trainloader):
            steps+=1
            if gpu==True:
                i,l=i.to('cuda'),l.to('cuda')
            optimizer.zero_grad()
            outputs=model.forward(i)
            loss=criterion(outputs,l)
            loss.backward()
            optimizer.step()
            run_loss+=loss.item()
            
            if steps % print_every==0:
                model.eval()
                losval,accuracy=valid(criterion,model,validloader)
                print("Epoch:{}/{}".format(e+1,epochs),"... Loss:{}".format(run_loss/print_every),"Validation Loss:{}".format(losval/le),
                      "Validation Accuracy:{}".format(accuracy))
                run_loss=0
                model.train()
        e+=1

def accuracy(testloader,model,criterion,gpu):
    right=0
    allto=0
    test_loss=0
    with torch.no_grad():
        for d in testloader:
            i,l=d
            if gpu==True:
                i,l=i.to('cuda'),l.to('cuda')
            output=model.forward(i)
            test_loss+=criterion(output,l).item()
            
            #flower index
            prob=torch.exp(output)
            pred=prob.max(dim=1)
            matches=(pred[1]==l.d)
            right+=matches.sum().item()
            allto+=64
        goal=100*(right/allto)
        return goal

    #valid def
def valid(criterion,model,validloader):    
    right=0
    allto=0
    val_loss=0
    with torch.no_grad():
        for d in validloader:
            i,l=d
            if gpu==True:
                i,l=i.to('cuda'),l.to('cuda')
            output=model.forward(i)
            val_loss+=criterion(output,l).item()
            
            #flower index
            prob=torch.exp(output)
            pred=prob.max(dim=1)
            matches=(pred[1]==l.d)
            right+=matches.sum().item()
            allto+=64
        goal=100*(right/allto)
        return goal
    
 #run
if __name__ == "__main__":
    main()