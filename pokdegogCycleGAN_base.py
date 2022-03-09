import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
import numpy as np
import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import color
import numpy as np
torch.manual_seed(0)

# Custom Dataloader
class ImageDataset(data.Dataset):
    def __init__(self, data_rootA, data_rootB, transform=None):
        self.transform = transform
        self.files_A = []
        self.files_B = []
        for path, subdirs, files in os.walk(data_rootA):
            for name in files:
                if name.endswith(".jpg"):
                    self.files_A.append(os.path.join(path, name))
        for path, subdirs, files in os.walk(data_rootB):
            for name in files:
                if name.endswith(".png"):
                    self.files_B.append(os.path.join(path, name))
        if len(self.files_A) > len(self.files_B):
            self.files_A, self.files_B = self.files_B, self.files_A
        self.new_perm()
        assert len(self.files_A) > 0, "Make sure you downloaded the images!"

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[self.randperm[index]]))
        if index == len(self) - 1:
            self.new_perm()
        return (item_A - 0.5) * 2, (item_B - 0.5) * 2

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))

#define criterion, adv_criterion -> adversarial loss, recon_crtiterion -> cyle consistency loss, idenity loss
adv_criterion = nn.MSELoss() 
recon_criterion = nn.L1Loss() 

#define hyper parameters
n_epochs = 1
dim_A = 3
dim_B = 3
display_step = 1000
batch_size = 2
lr = 0.0001
load_shape = 286
target_shape = 128
device = 'cuda'

num_workers = 2. # how many subprocesses to use for data loading. these work asynchronously on CPU as your model works on GPU

#create custom transform
transform = transforms.Compose([
    transforms.Lambda(lambda img: transforms.ToTensor()(img)[:3,:, :]), #ensure the image is only 3 channels
    transforms.Resize(size = (target_shape,target_shape)),              #set the image to the deisred shape
    transforms.RandomHorizontalFlip(),                                  #random flip
    transforms.Normalize(mean=(0.5,),std=(0.5,)),                       #normalize for faster convergence
])

#build dataloader
dataset = ImageDataset(data_rootA="Downloads/Images", data_rootB ="Gproj", transform=transform)
dataloader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True)

#define different block structures for network architecture
class ResidualBlock(nn.Module):
    '''
    ResidualBlock Class:
    Performs two convolutions and an instance normalization, the input is added
    to this output to form the residual block output.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.instancenorm = nn.InstanceNorm2d(input_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        Function for completing a forward pass of ResidualBlock: 
        Given an image tensor, completes a residual block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        original_x = x.clone()
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instancenorm(x)
        return original_x + x

class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs a convolution followed by a max pool operation and an optional instance norm.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, padding=1, stride=2, padding_mode='reflect')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock: 
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class ExpandingBlock(nn.Module):
    '''
    ExpandingBlock Class:
    Performs a convolutional transpose operation in order to upsample, 
        with an optional instance norm
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        Function for completing a forward pass of ExpandingBlock: 
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a Generator - 
    maps each the output to the desired number of output channels
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock: 
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x

#define Generator structure
class Generator(nn.Module):
    '''
    Generator Class
    A series of 2 contracting blocks, 9 residual blocks, and 2 expanding blocks to 
    transform an input image into an image from the other class, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(Generator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        res_mult = 4
        self.res0 = ResidualBlock(hidden_channels * res_mult)
        self.res1 = ResidualBlock(hidden_channels * res_mult)
        self.res2 = ResidualBlock(hidden_channels * res_mult)
        self.res3 = ResidualBlock(hidden_channels * res_mult)
        self.res4 = ResidualBlock(hidden_channels * res_mult)
        self.res5 = ResidualBlock(hidden_channels * res_mult)
        self.res6 = ResidualBlock(hidden_channels * res_mult)
        self.res7 = ResidualBlock(hidden_channels * res_mult)
        self.res8 = ResidualBlock(hidden_channels * res_mult)
        self.expand2 = ExpandingBlock(hidden_channels * 4)
        self.expand3 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        '''
        Function for completing a forward pass of Generator: 
        Given an image tensor, passes it through the U-Net with residual blocks
        and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        x9 = self.res6(x8)
        x10 = self.res7(x9)
        x11 = self.res8(x10)
        x12 = self.expand2(x11)
        x13 = self.expand3(x12)
        xn = self.downfeature(x13)
        return self.tanh(xn)

#define Discriminator structure
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=4, activation='lrelu')
        self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=4, activation='lrelu')
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn

#build Generators and Discriminators
gen_AB = Generator(dim_A, dim_B).to(device)
gen_BA = Generator(dim_B, dim_A).to(device)
gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=lr, betas=(0.5, 0.999))
disc_A = Discriminator(dim_A).to(device)
disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=lr, betas=(0.5, 0.999))
disc_B = Discriminator(dim_B).to(device)
disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=lr, betas=(0.5, 0.999))

#create function to initialize network weights
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

#optional: load model as pretrained starting point
pretrained = False
if pretrained:
    pre_dict = torch.load('name.pth')
    gen_AB.load_state_dict(pre_dict['gen_AB'])
    gen_BA.load_state_dict(pre_dict['gen_BA'])
    gen_opt.load_state_dict(pre_dict['gen_opt'])
    disc_A.load_state_dict(pre_dict['disc_A'])
    disc_A_opt.load_state_dict(pre_dict['disc_A_opt'])
    disc_B.load_state_dict(pre_dict['disc_B'])
    disc_B_opt.load_state_dict(pre_dict['disc_B_opt'])
else:
    gen_AB = gen_AB.apply(weights_init)
    gen_BA = gen_BA.apply(weights_init)
    disc_A = disc_A.apply(weights_init)
    disc_B = disc_B.apply(weights_init)

#class loss function(penalizes a discriminator for the difference in its prediction of the opposite class and a fake image generated from that class) - see paper for details
def get_class_loss(real_Y, fake_X, disc_X):
    pred_r = disc_X(real_Y)
    pred_f = disc_X(fake_X)
    return nn.L1Loss()(pred_r, pred_f) 

#modified get_disc_loss function inclusive of the class loss
def get_disc_loss_class(real_X, fake_X, real_Y, disc_X, adv_criterion):
    pred = disc_X(real_X)
    real_loss = adv_criterion(pred, torch.ones_like(pred))
    pred =  disc_X(fake_X)
    fake_loss = adv_criterion(pred, torch.zeros_like(pred))
    class_loss = get_class_loss(real_Y, fake_X, disc_X)
    disc_loss = (real_loss + fake_loss)/2 + class_loss 
    return disc_loss

#return the adversarial loss for a given discriminator
def get_disc_loss(real_X, fake_X, disc_X, adv_criterion):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        real_X: the real images from pile X
        fake_X: the generated images of class X
        disc_X: the discriminator for class X; takes images and returns real/fake class X
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator 
            predictions and the target labels and returns a adversarial 
            loss (which you aim to minimize)
    '''
    pred = disc_X(real_X)
    real_loss = adv_criterion(pred, torch.ones_like(pred))
    pred =  disc_X(fake_X)
    fake_loss = adv_criterion(pred, torch.zeros_like(pred))
    disc_loss = (real_loss + fake_loss)/2
    return disc_loss

#return the adversarial loss of a generator
def get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion):
    '''
    Return the adversarial loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        disc_Y: the discriminator for class Y; takes images and returns real/fake class Y
            prediction matrices
        gen_XY: the generator for class X to Y; takes images and returns the images 
            transformed to class Y
        adv_criterion: the adversarial loss function; takes the discriminator 
                  predictions and the target labels and returns a adversarial 
                  loss (which you aim to minimize)
    '''
    fake_Y = gen_XY(real_X)
    pred = disc_Y(fake_Y)
    adversarial_loss = adv_criterion(pred, torch.ones_like(pred))
    return adversarial_loss, fake_Y

#return the identity loss for a generator using the identity_criterion
def get_identity_loss(real_X, gen_YX, identity_criterion):
    '''
    Return the identity loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        gen_YX: the generator for class Y to X; takes images and returns the images 
            transformed to class X
        identity_criterion: the identity loss function; takes the real images from X and
                        those images put through a Y->X generator and returns the identity 
                        loss (which you aim to minimize)
    '''
    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(real_X, identity_X)
    return identity_loss, identity_X

#alternative identity loss function - return the identity loss for a generator using the MSELoss between discriminator prediction
def get_identity_loss_disc(real_X, gen_YX, disc_X):
    identity_X = gen_YX(real_X)
    pred_r = disc_X(real_X)
    pred_f = disc_X(identity_X)
    identity_loss =  nn.MSELoss()(pred_r, pred_f)
    return identity_loss, identity_X

#return the cycle consisitency loss for the generators using the cycle_criterion
def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    '''
    Return the cycle consistency loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        fake_Y: the generated images of class Y
        gen_YX: the generator for class Y to X; takes images and returns the images 
            transformed to class X
        cycle_criterion: the cycle consistency loss function; takes the real images from X and
                        those images put through a X->Y generator and then Y->X generator
                        and returns the cycle consistency loss (which you aim to minimize)
    '''
    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(real_X,cycle_X)
    return cycle_loss, cycle_X

#alternative cycle consistency loss function- return the cycle consisitency loss for the generators using the MSELoss between discriminator prediction
def get_cycle_consistency_loss_disc(real_X, fake_Y, gen_YX, disc_X):
    cycle_X = gen_YX(fake_Y)
    pred_r = disc_X(real_X)
    pred_f = disc_X(cycle_X)
    cycle_loss =  nn.MSELoss()(pred_r, pred_f)
    return cycle_loss, cycle_X

#returns the Diff Loss for a generator - see paper for details  
def get_diff_loss(real_X, gen_XY):
    fake_Y = gen_XY(real_X)
    return torch.square(torch.nn.L1Loss()(real_X, fake_Y)) / torch.nn.MSELoss()(real_X, fake_Y)

#return the retention loss for a generator - see paper for details
def retention_loss(fake_Y, disc_X):
    pred = disc_X(fake_Y)
    return adv_criterion(pred, torch.zeros_like(pred))

#get the overall gen_loss
def get_gen_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10, lambda_diff = 0.5, lambda_ret = 0.5):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        real_A: the real images from pile A
        real_B: the real images from pile B
        gen_AB: the generator for class A to B; takes images and returns the images 
            transformed to class B
        gen_BA: the generator for class B to A; takes images and returns the images 
            transformed to class A
        disc_A: the discriminator for class A; takes images and returns real/fake class A
            prediction matrices
        disc_B: the discriminator for class B; takes images and returns real/fake class B
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator 
            predictions and the true labels and returns a adversarial 
            loss (which you aim to minimize)
        identity_criterion: the reconstruction loss function used for identity loss
            and cycle consistency loss; takes two sets of images and returns
            their pixel differences (which you aim to minimize)
        cycle_criterion: the cycle consistency loss function; takes the real images from X and
            those images put through a X->Y generator and then Y->X generator
            and returns the cycle consistency loss (which you aim to minimize).
            Note that in practice, cycle_criterion == identity_criterion == L1 loss
        lambda_identity: the weight of the identity loss
        lambda_cycle: the weight of the cycle-consistency loss
        lambda_diff: the weight for the Diff Loss
        lambda_ret: the weight for the retention loss
    '''
    adv_lossA, fake_B = get_gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion)
    adv_lossB, fake_A = get_gen_adversarial_loss(real_B, disc_A, gen_BA, adv_criterion)
    adv_loss = adv_lossA + adv_lossB
    id_loss = get_identity_loss(real_A, gen_BA, identity_criterion)[0] + get_identity_loss(real_B, gen_AB, identity_criterion)[0]
    #id_loss = get_identity_loss_disc(real_A, gen_BA, disc_A)[0] + get_identity_loss_disc(real_B, gen_AB, disc_B)[0] #<-uncomment to use alternative identity loss, comment the line above
    cyc_loss =  get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)[0] + get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)[0]
    #cyc_loss =  get_cycle_consistency_loss_disc(real_B, fake_A, gen_AB, disc_B)[0] + get_cycle_consistency_loss_disc(real_A, fake_B, gen_BA, disc_A)[0] #<-uncomment to use alternative cycle loss, comment the line above
    diff_loss = get_diff_loss(real_A, gen_AB) + get_diff_loss(real_B, gen_BA)
    #ret_loss = retention_loss(fake_B, disc_A) + retention_loss(fake_A, disc_B) #<-uncomment to use retentiton loss, recommend removing diff loss if using retention loss
    # Total loss - sum all loss terms using the appropriate lambda
    gen_loss = adv_loss + (lambda_identity*id_loss) + (lambda_cycle*cyc_loss) + (lambda_diff * diff_loss)
    return gen_loss, fake_A, fake_B

plt.rcParams["figure.figsize"] = (10, 10)

#train model
def train(save_model=False):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    cur_step = 0

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for real_A, real_B in tqdm(dataloader):
            gray = transforms.RandomGrayscale(p= 0.25 * np.exp(-cur_step/len(real_A))) #random decaying grayscaling
            real_A = gray(real_A) #comment this and below line to not use grayscaling
            real_B = gray(real_B) #comment this and above line to not use grayscaling
            real_A = nn.functional.interpolate(real_A, size=target_shape)
            real_B = nn.functional.interpolate(real_B, size=target_shape)
            cur_batch_size = len(real_A)
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            ### Update discriminator A ###
            disc_A_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_A = gen_BA(real_B)
            disc_A_loss = get_disc_loss(real_A, fake_A, disc_A, adv_criterion)
            #disc_A_loss = get_disc_loss_class(real_A, fake_A, real_B, disc_A, adv_criterion) #<- uncomment to use alternative discriminator loss, comment line above
            disc_A_loss.backward(retain_graph=True) # Update gradients
            disc_A_opt.step() # Update optimizer

            ### Update discriminator B ###
            disc_B_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_B = gen_AB(real_A)
            disc_B_loss = get_disc_loss(real_B, fake_B, disc_B, adv_criterion)
            #disc_B_loss = get_disc_loss_class(real_B, fake_B, real_A, disc_B, adv_criterion) #<- uncomment to use alternative discriminator loss, comment line above
            disc_B_loss.backward(retain_graph=True) # Update gradients
            disc_B_opt.step() # Update optimizer

            ### Update generator ###
            gen_opt.zero_grad()
            gen_loss, fake_A, fake_B = get_gen_loss(
                real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, recon_criterion, recon_criterion
            )
            gen_loss.backward() # Update gradients
            gen_opt.step() # Update optimizer

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_A_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            ### Visualization code ###
            if cur_step % display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                torch.save({
                        'gen_AB': gen_AB.state_dict(),
                        'gen_BA': gen_BA.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc_A': disc_A.state_dict(),
                        'disc_A_opt': disc_A_opt.state_dict(),
                        'disc_B': disc_B.state_dict(),
                        'disc_B_opt': disc_B_opt.state_dict()
                }, f"name_{cur_step}.pth")
            cur_step += 1
train()