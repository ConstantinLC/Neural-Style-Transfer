import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from PIL import Image

import torchvision
from torchvision import transforms, datasets, utils

from google.colab import files

import warnings
warnings.filterwarnings("ignore")

class neural_style_transfer():
  
  def __init__(self, content, style, generated):
    self.content = content
    self.style = style
    self.generated = generated.requires_grad_()
    
    self.content_layers = [3] #Conv layers used in content loss
    self.style_layers = [1, 2, 3, 4, 5] #Conv layers used in style loss
    
    self.content_weight = 1
    self.style_weight = 1000000
    
    self.cnn = torchvision.models.vgg19(pretrained = True).cuda()
    
    self.criterion = nn.MSELoss()
    
    self.content_list = []
    self.style_list = []
    
    self.optimizer = torch.optim.LBFGS([self.generated])

  def gram_matrix(self, x):
    '''Computes the gram matrix, where element (i,j) is dot product between i-th and j-th filters in input image'''
    y = x.squeeze(0)
    N, M1, M2 = y.size()
    M = M1*M2
    y = y.view(N, M)
    return y.matmul(y.t())/(2*N*M) #Same as dividing the loss by 4*N^2*M^2
  
  def activations_content_style(self):
    '''Precomputes the activations of content and style images to avoid doing it at every training loop'''
    
    content = self.content
    style = self.style
    
    conv_i = 1 #Index of current conv layer
    for l in self.cnn.features:
      
      if isinstance(l, nn.ReLU):
        l = nn.ReLU(inplace = False)
      l.cuda()
      
      content = l.forward(content)
      style = l.forward(style)
      
      if isinstance(l, nn.Conv2d):
        
        if conv_i in self.content_layers :
          self.content_list.append(content)
        
        if conv_i in self.style_layers :
          self.style_list.append(style)
          
        conv_i += 1
        
  def optimize_step(self):
    '''Computes the loss and does one step of L-BFGS algorithm (paper by Gatis uses this method)'''
    
    self.generated.data.clamp_(0, 1) #keep pixel values between 0 and 1. variable.data.clamp() is different from variable.clamp() !
    
    def closure(): #LBFGS computes the loss multiple times in a step
      
      self.optimizer.zero_grad()
      generated = self.generated
      
      loss_content = 0
      loss_style = 0

      conv_i = 1
      content_i = 0
      style_i = 0
      
      for l in self.cnn.features:

        if isinstance(l, nn.ReLU):
          l = nn.ReLU(inplace = False)
        l.cuda()
        
        generated = l.forward(generated)
        
        if isinstance(l, nn.Conv2d):
          
          if conv_i in self.content_layers :
            loss_content += self.criterion(generated, self.content_list[content_i].detach()) 
            '''detach is necessary here since it avoids connections between computation graphs at different steps'''
            content_i += 1
          
          if conv_i in self.style_layers :
            loss_style += self.criterion(self.gram_matrix(generated), self.gram_matrix(self.style_list[style_i].detach()))
            style_i += 1
            
          conv_i += 1
        
      total_loss = loss_content * self.content_weight + loss_style * self.style_weight
      total_loss.backward()
      return total_loss
    
    self.optimizer.step(closure)
    
def load_image(path):
  #Taken from Pytorch tutorial https://pytorch.org/tutorials/index.html
  imsize = (256, 256)
  loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()]) #VGG19 expects a 256*256 image
  """load image, returns cuda tensor"""
  image = Image.open(path)
  image = loader(image).unsqueeze(0)
  return image

def main():
  
  nb_iterations = 20

  style = load_image('/content/gdrive/My Drive/Colab Notebooks/Neural Style Transfer/munch.jpg').float().cuda()
  content = load_image('/content/gdrive/My Drive/Colab Notebooks/Neural Style Transfer/iowa.jpg').float().cuda()
  white_noise = load_image('/content/gdrive/My Drive/Colab Notebooks/Neural Style Transfer/white_noise.png').float().cuda()
  white_noise = torch.cat((white_noise, white_noise, white_noise), dim = 1) #white_noise is a black and white image
  
  style_transfer = neural_style_transfer(content, style, white_noise)
  style_transfer.activations_content_style()
  
  for i in range(nb_iterations):
    style_transfer.optimize_step()
    print("Step", i + 1)
    
  print("Training done")
  generated = style_transfer.generated
  plt.imshow(generated.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
  torchvision.utils.save_image(generated, '/content/gdrive/My Drive/Colab Notebooks/Neural Style Transfer/iowa_generated.png')
  
main()