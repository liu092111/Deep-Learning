#import package
import torch
import torch.nn as nn
import torch.optim as optim 

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torchvision.models import vgg19, VGG19_Weights

#Configuration
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

imsize=512

style_path='D:\資料備份-勿刪\Desktop\碩一修課\Deep Learning\Generative and Neural Style Transfer\data\images\style\Zodiac.jpg'
content_path='D:\資料備份-勿刪\Desktop\碩一修課\Deep Learning\Generative and Neural Style Transfer\data\images\content\content_path.jpg'

content_layer_default=['conv_3', 'conv_4']
style_layer_default=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5','conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10']

num_run=2000
style_weight=50000
content_weight=1

#Loading data
#Original Image
style_img=Image.open(style_path)
content_img=Image.open(content_path)

plt.figure(figsize=(8,6))
plt.axis('off')
plt.title('Original figure')

plt.subplot(1,2,1)
plt.imshow(style_img)
#plt.axis('off')
plt.title('Style image')

plt.subplot(1,2,2)
plt.imshow(content_img)
#plt.axis('off')
plt.title('Content image')
plt.show()

loader=transforms.Compose([transforms.Resize([imsize,imsize]), transforms.ToTensor()])

def image_loader(img_path):
    img=Image.open(img_path)
    img=loader(img).unsqueeze(0)
    return img

style_img=image_loader(style_path).to(device, torch.float)
content_img=image_loader(content_path).to(device, torch.float)

#assert style_img.size() == content_img.size()

#Resize Image
unloader=transforms.ToPILImage()
style_img_o=unloader(style_img.squeeze(0))
content_img_o=unloader(content_img.squeeze(0))

plt.figure(figsize=(8,6))
plt.title('Resize figure')
plt.axis('off')

plt.subplot(1,2,1)
plt.imshow(style_img_o)
plt.title('Style image')

plt.subplot(1,2,2)
plt.imshow(content_img_o)
plt.title('Content image')

plt.show()

#Loss Function
#Contain Loss
class Contentloss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target=target.detach()
        self.loss_fn=nn.MSELoss()
        
    def forward(self, input):
        self.loss=self.loss_fn(input, self.target)
        return input
    
#Style Loss
def gram_matrix(input):
    a, b, c, d=input.size()
    features=input.view(a*b,c*d)
    G=torch.mm(features, features.t())
    return G.div(a*b*c*d)

class Styleloss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target=gram_matrix(target_feature).detach()
        self.loss_fn=nn.MSELoss()
    def forward(self, input):
        G=gram_matrix(input)
        self.loss=self.loss_fn(self.target, G)
        return input

#Import Module
cnn=vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

#Normalization
cnn_normalization_mean=torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std=torch.tensor([0.229, 0.224, 0.225])

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean=mean.view(-1, 1, 1)
        self.std=std.view(-1,1,1)
    def forward(self,input):
        output=(input-self.mean)/self.std
        return output

#Getting and adding layers from VGG19
normalize=Normalization(cnn_normalization_mean, cnn_normalization_std)
model=nn.Sequential(normalize)
model.to(device)

def get_layer_and_add_loss(cnn, model, content_img, style_img, content_layer, style_layer):
    i=0
    content_losses=[]
    style_losses=[]
    
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i+=1
            name='conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name='relu_{}'.format(i)
            layer=nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name='pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name='bl_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer : {}'.format(layer.__class__.__name__))

        model.add_module(name,layer)
        model.to(device)
        
        if name in content_layer:
            target=model(content_img).detach()
            contentloss=Contentloss(target)
            model.add_module('content_loss_{}'.format(i), contentloss)
            content_losses.append(contentloss)
        if name in style_layer:
            print('i:{}, style:{}'.format(i,style_img.size()))
            target_feature=model(style_img).detach()
            styleloss=Styleloss(target_feature)
            model.add_module('style_loss_{}'.format(i), styleloss)
            style_losses.append(styleloss)

    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], Contentloss) or isinstance(model[i], Styleloss):
            break
    model=model[:(i+1)].to(device)
    return model, content_losses, style_losses

#Input Image(White Noise)
input_img=torch.randn(content_img.size())
input_medium=input_img.cpu()
plt.figure()
plt.imshow(input_medium.squeeze(0).permute(1,2,0))
plt.title('Input image')
plt.show()

input_img.size()

#Optimizer(Gradient descent)
def optimizer_input_img(input_img):
    optimizer=optim.LBFGS([input_img])
    return optimizer

#Generation Loop
def run_model(cnn, model,content_img,style_img,content_layer,style_layer,input_img,num_run,content_weight,style_weight):

    print('Building neural transfer model.')
    model, content_losses, style_losses=get_layer_and_add_loss(cnn,model,content_img,style_img,content_layer,style_layer)

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer=optimizer_input_img(input_img)
    print('Optimizing')

    run=[0]

    while run[0]<=num_run:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0,1)

            optimizer.zero_grad()
            model(input_img)
            content_score=0
            style_score=0

            for cl in content_losses:
                content_score+=cl.loss
            for sl in style_losses:
                style_score+=sl.loss

            content_score=content_score*content_weight
            style_score=style_score*style_weight
            loss=content_score+style_score

            loss.backward()
            
            run[0]+=1

            if run[0]%50==0:
                print('run: {}'.format(run))
                print('Content loss: {}, Style loss: {}'.format(content_score, style_score))

                input_medium=input_img.squeeze(0).cpu().detach()
                plt.figure()
                plt.ioff()
                plt.imshow(input_medium.permute(1,2,0))
                plt.title('run: {}'.format(run))

                save_path = fr'D:\資料備份-勿刪\Desktop\碩一修課\Deep Learning\Generative and Neural Style Transfer\{run[0]}.png'
                plt.savefig(save_path)  # 存檔
                plt.show(block = False)
        
            return content_score+style_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp(0,1)
    return input_img

#Image generation
output_img=run_model(cnn, model,content_img,style_img,content_layer_default,style_layer_default,input_img,num_run,content_weight,style_weight)

model