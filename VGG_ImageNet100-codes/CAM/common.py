'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''


'''
Common functions for simple PyTorch MNIST example
'''

'''
Author: Mark Harvey, Xilinx inc
'''


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys
import torch
import torch.nn as nn
import time
#from .utils import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast


from cbam import *
from bam import *

class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 100,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    a = 0
    for v in cfg:
        a = a + 1
        if a == 16:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=(1, 1))
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True),]
            in_channels = v
        elif a == 17:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=(1, 1))
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True), nn.Dropout()]
        elif v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            
    # conv2d = nn.Conv2d(512, 4096, kernel_size=(7, 7), stride=(1, 1))
    # # nn.ReLU(True),
    # # nn.Dropout(),
    # layers += [conv2d, nn.ReLU(inplace=True),nn.Dropout()]
    # conv2d = nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1))
    # # nn.ReLU(True),
    # # nn.Dropout(),
    # layers += [conv2d, nn.ReLU(inplace=True),nn.Dropout()]
    conv2d = nn.Conv2d(cfg[-1], 100, kernel_size=(1, 1), stride=(1, 1))
    # nn.Flatten()
    layers += [conv2d,nn.Flatten()]
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


#def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
def vgg(arch: str, cfg: List[Union[str, int]], batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg, batch_norm=batch_norm), **kwargs)
    #model = nn.Sequential(nn.Sequential())
    return model


class fusedpool(nn.Module):
    def __init__(self,shape):
        
        super(fusedpool,self).__init__()
        
        self.max_pool = nn.MaxPool2d(2, 2, padding=0, dilation=1, ceil_mode=False)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((shape,shape))
        
        
        
    def forward(self,x):
        
        b,c,_,_ = x.size()
        
        x1 = self.max_pool(x)#.view(b,c)
        
        x2 = self.avg_pool(x)#.view(b,c)
        
        x3= torch.cat([x1,x2],1)
        
        return x3
        
    
    
class CNN(nn.Module):
    def __init__(self, num_classes = 100,init_weights=True):
        super(CNN,self).__init__()
        
        
        self.features1 = nn.Sequential(
            
                  nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True),
                  
                  nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True),
                  
                  CBAM(64),
                  
                  )
        
        self.fusepool1 = fusedpool(112)

        self.features2 = nn.Sequential(
            
                  #nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1)),
            
                  nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True),
                  
                  
                  nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True),
                  
                  CBAM(64),
                  
                   )
        self.fusepool2 = fusedpool(56)
        
        self.features3 = nn.Sequential(
                  
                  nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.BatchNorm2d(128),
                  nn.ReLU(True),
                  
                   )
        self.fusepool3 = fusedpool(28)    
        
        self.features4 = nn.Sequential(
            
                  #nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
                  
                  nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.BatchNorm2d(128),
                  nn.ReLU(True),
                  
                  CBAM(128),
                  
                   )
        self.fusepool4 = fusedpool(14) 
        
        
        self.features5 = nn.Sequential(
          
          #nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
          
          nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  
          nn.BatchNorm2d(128),
          nn.ReLU(True),
          
          CBAM(128),
          
          
           )
        self.fusepool5 = fusedpool(7)  
        
        self.features6 = nn.Sequential(
            
            #nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
            
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.AdaptiveAvgPool2d((2, 2)),
            
            nn.Flatten(),

            nn.Dropout(),
            nn.Linear(128*2*2, 100),
      
            )

        if init_weights:
              self._initialize_weights()

    def forward(self,x):
        x = self.features1(x)
        x = self.fusepool1(x)
        
        x = self.features2(x)
        x = self.fusepool2(x)
        
        x = self.features3(x)
        x = self.fusepool3(x)
        
        x = self.features4(x)
        x = self.fusepool4(x)
        
        x = self.features5(x)
        x = self.fusepool5(x)
 
        x = self.features6(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.5)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                

def train(model, device, train_loader, optimizer, epoch):
    '''
    train the model
    '''
    model.train()
    

    counter = 0
    correct1 = 0
    print("Epoch "+str(epoch))
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        x = model(data)
        output = F.log_softmax(input=x,dim=1)
        pred = output.argmax(dim=1, keepdim=True)
        correct1 += pred.eq(target.view_as(pred)).sum().item()
        loss = F.nll_loss(output, target)
        loss.backward()
        #updateBN(model)
        optimizer.step()
        counter += 1
    acc1 = 100. * correct1 / len(train_loader.dataset)
    print('\Train set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct1, len(train_loader.dataset), acc1))
    return acc1,loss.item()


global time_all


def test(model, device, test_loader):
    '''
    test the model
    '''
    model.eval()
    #test_loss = 0
    time_all = 0
    correct = 0

    with torch.no_grad():
        
        #for batch_idx1, (data, target) in enumerate(test_loader):
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            time1 = time.time()
            
            y = model(data)
            
            #sys.exit()
            
            time2 = time.time()
            
            time_1 = time2-time1
            
            time_all = time_all + time_1
            
            #print(time2-time1)
        
            output = F.log_softmax(input=y,dim=1)
            test_loss = F.nll_loss(output, target)
            pred1 = output.argmax(dim=1, keepdim=True)
            correct += pred1.eq(target.view_as(pred1)).sum().item()
            #break
    print(time_all)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_loader.dataset), acc))

    return acc,test_loss.item()




