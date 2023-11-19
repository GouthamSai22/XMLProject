import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

DEBUG = False

class StudentConceptizer(nn.Module):
    def __init__(self, input_dim, concept_num, concept_dim=None, num_channel=1):
        super(StudentConceptizer, self).__init__()
        self.input_dim = input_dim
        self.concept_num = concept_num
        self.concept_dim = concept_dim
        self.num_channel = num_channel
        self.learnable = True
        self.add_bias = False
        # Computing the output Dimensions is necessary - Figure it out
        self.output_dim = int(np.sqrt(input_dim)//4 - 3*(5-1)//4)

        # Encoding
        self.conv1  = nn.Conv2d(num_channel, 10, kernel_size=5)
        self.conv2  = nn.Conv2d(10, concept_num, kernel_size=5)
        self.linear = nn.Linear(self.output_dim**2, self.concept_dim)

        # Decoding
        self.unlinear = nn.Linear(self.concept_dim, self.output_dim**2)
        self.deconv3  = nn.ConvTranspose2d(self.concept_num, 16, 5, stride = 2)
        self.deconv2  = nn.ConvTranspose2d(16, 8, 5)
        self.deconv1  = nn.ConvTranspose2d(8, num_channel, 2, stride=2, padding=1)

    def encode(self, x):
        p = F.relu(F.max_pool2d(self.conv1(x), 2))
        p = F.relu(F.max_pool2d(self.conv2(p), 2))
        encoded = self.linear(p.view(-1, self.concept_num, self.output_dim**2))
        return encoded

    def decode(self, z):
        q = self.unlinear(z)
        q = q.view(-1, self.concept_num, self.output_dim, self.output_dim)
        q = F.relu(self.deconv3(q))
        q = F.relu(self.deconv2(q))
        decoded = F.tanh(self.deconv1(q))
        return decoded
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded.view_as(x)

    
class StudentParametrizer(nn.Module):
    def __init__(self, input_dim, concept_num, output_dim, num_channel=1, only_positive=False):
        super(StudentParametrizer, self).__init__()
        self.concept_num = concept_num
        self.output_dim = output_dim
        self.input_dim  = input_dim
        
        self.conv1 = nn.Conv2d(num_channel, 10, kernel_size=5) 
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.output_dim_conv = int(np.sqrt(input_dim)//4 - 3*(5-1)//4)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20*(self.output_dim_conv**2), concept_num*output_dim)
        self.positive = only_positive

    def forward(self, x):
        p = F.relu(F.max_pool2d(self.conv1(x), 2))
        p = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(p)), 2))
        p = p.view(-1, 20*(self.output_dim_conv**2))
        p = self.fc1(p)
        out = F.dropout(p, training=self.training).view(-1,self.concept_num,self.output_dim)
        return out


class StudentAggregator(nn.Module):
    def __init__(self, concept_dim, num_classes):
        super(StudentAggregator, self).__init__()
        self.concept_dim = concept_dim 
        self.num_classes = num_classes

    def forward(self, H, Th):
        assert H.size(-2) == Th.size(-2), "Number of concepts in H and Th don't match"
        assert H.size(-1) == 1, "Concept h_i should be scalar, not vector sized"
        assert Th.size(-1) == self.num_classes, "Wrong Theta size"
        out = torch.bmm(Th.transpose(1,2), H).squeeze(dim=-1)
        return out


class StudentGSENN(nn.Module):
    ''' Wrapper for GSENN with H-learning'''

    def __init__(self, conceptizer, parametrizer, aggregator):
        super(StudentGSENN, self).__init__()
        self.conceptizer = conceptizer
        self.parametrizer = parametrizer
        self.aggregator = aggregator
        self.learning_H = conceptizer.learnable

    def forward(self, x):
        if DEBUG:
            print('Input to GSENN:', x.size())

        if self.learning_H:
            h_x, x_tilde = self.conceptizer(x)
            self.recons = x_tilde
            
            self.h_norm_l1 = h_x.norm(p=1)
        else:
            h_x = self.conceptizer( x.detach().requires_grad_(False) )

        self.concepts = h_x

        if DEBUG:
            print('Encoded concepts: ', h_x.size())
            if self.learning_H:
                print('Decoded concepts: ', x_tilde.size())


        thetas = self.parametrizer(x)
        if self.parametrizer.positive:
            thetas = F.sigmoid(thetas)
        else:
            thetas = F.tanh(thetas)

        if len(thetas.size()) == 2:
            thetas = thetas.unsqueeze(2)

        # Store local Parameters
        self.thetas = thetas

        if DEBUG:
            print('Theta: ', thetas.size())

        if len(h_x.size()) == 4:
            # Concepts are two-dimensional, so flatten
            h_x = h_x.view(h_x.size(0), h_x.size(1), -1)


        out = self.aggregator(h_x, thetas)

        if DEBUG:
            print('Output: ', out.size())

        return out