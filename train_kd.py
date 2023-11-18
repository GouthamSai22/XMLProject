import numpy as np

# Torch Imports
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.dataloader as dataloader

#local imports
from models.teacher import Conceptizer, Parametrizer, Aggregator, GSENN
from models.student import StudentConceptizer, StudentParametrizer, StudentAggregator, StudentGSENN
from explainability.explanations import show_explainations, show_prototypes, generate_explantions
from trainer import ClassificationTrainer

def load_mnist_data(valid_size=0.1, shuffle=True, random_seed=2008, batch_size = 64, num_workers = 1):
    """
        We return train and test for plots and post-training experiments
    """
    transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

    train = MNIST('data/MNIST', train=True, download=True, transform=transform)
    test  = MNIST('data/MNIST', train=False, download=True, transform=transform)

    num_train = len(train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)


    # Create DataLoader
    dataloader_args = dict(batch_size=batch_size,num_workers=num_workers)
    train_loader = dataloader.DataLoader(train, sampler=train_sampler, **dataloader_args)
    valid_loader = dataloader.DataLoader(train, sampler=valid_sampler, **dataloader_args)
    dataloader_args['shuffle'] = True
    test_loader = dataloader.DataLoader(test, **dataloader_args)

    return train_loader, valid_loader, test_loader, train, test


def load_cifar_data(valid_size=0.1, shuffle=True, resize = None, random_seed=2008, batch_size = 64, num_workers = 1):
    """
        We return train and test for plots and post-training experiments
    """
    transf_seq = [
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if resize and (resize[0] != 32 or resize[1] != 32):
        transf_seq.insert(0, transforms.Resize(resize))

    transform = transforms.Compose(transf_seq)
    # normalized according to pytorch torchvision guidelines https://chsasank.github.io/vision/models.html
    train = CIFAR10('data/CIFAR', train=True, download=True, transform=transform)
    test  = CIFAR10('data/CIFAR', train=False, download=True, transform=transform)

    num_train = len(train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)


    # Create DataLoader
    dataloader_args = dict(batch_size=batch_size,num_workers=num_workers)
    train_loader = dataloader.DataLoader(train, sampler=train_sampler, **dataloader_args)
    valid_loader = dataloader.DataLoader(train, sampler=valid_sampler, **dataloader_args)
    dataloader_args['shuffle'] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)

    return train_loader, valid_loader, test_loader, train, test

def main():
    np.random.seed(314)
    torch.manual_seed(314)
    
    epochs = 100
    batch_size = 128
    num_concepts = 12
    concept_dim = 1
    
    dataset = input ("Enter 0 for MNIST else CIFAR10 : ")
    
    if dataset == 0:
        num_classes = 10
        theta_dim = num_classes
        num_channels = 1
        H, W = 28, 28
        
        train_loader, valid_loader, test_loader, train_tds, test_tds = load_mnist_data(batch_size = batch_size, num_workers = 4)
    else:
        num_classes = 10
        theta_dim = num_classes
        num_channels = 3
        H, W = 32, 32
        
        train_loader, valid_loader, test_loader, train_tds, test_tds = load_cifar_data(batch_size = batch_size, num_workers = 4,resize = (H,W))
        
    
    input_dim = H*W
    
    conceptizer  = Conceptizer(input_dim, num_concepts, concept_dim, num_channel = num_channels)
    parametrizer = Parametrizer(input_dim, num_concepts, theta_dim, num_channel = num_channels)
    aggregator   = Aggregator(concept_dim, num_classes)
    teacher_model= GSENN (conceptizer, parametrizer, aggregator)
    
    trainer = ClassificationTrainer(teacher_model, num_classes)
    trainer.train(train_loader, val_loader = valid_loader, epochs = epochs, save_path = None)
    trainer.plot_losses(save_path = '/home/cs20btech11042/XML_KD/experiments/teacher')
    teacher_model.eval()
    teacher_accuracy = trainer.evaluate(test_loader)
    
    conceptizer  = StudentConceptizer(input_dim, num_concepts, concept_dim, num_channel = num_channels)
    parametrizer = StudentParametrizer(input_dim, num_concepts, theta_dim, num_channel = num_channels)
    aggregator   = StudentAggregator(concept_dim, num_classes)
    student_model= StudentGSENN (conceptizer, parametrizer, aggregator)
    
    trainer = ClassificationTrainer(student_model, num_classes)
    trainer.train(train_loader, val_loader = valid_loader, epochs = epochs, save_path = None)
    trainer.plot_losses(save_path = '/home/cs20btech11042/XML_KD/experiments/student')
    student_model.eval()
    student_accuracy = trainer.evaluate(test_loader)
    
    conceptizer  = StudentConceptizer(input_dim, num_concepts, concept_dim, num_channel = num_channels)
    parametrizer = StudentParametrizer(input_dim, num_concepts, theta_dim, num_channel = num_channels)
    aggregator   = StudentAggregator(concept_dim, num_classes)
    student_kd_model= StudentGSENN (conceptizer, parametrizer, aggregator)
    
    trainer = ClassificationTrainer(student_kd_model, num_classes)
    trainer.train(train_loader, val_loader = valid_loader, teacher_model = teacher_model, epochs = epochs, save_path = None)
    trainer.plot_losses(save_path = '/home/cs20btech11042/XML_KD/experiments/student_kd')
    student_kd_model.eval()
    student_kd_accuracy = trainer.evaluate(test_loader)
    
    conceptizer  = StudentConceptizer(input_dim, num_concepts, concept_dim, num_channel = num_channels)
    parametrizer = StudentParametrizer(input_dim, num_concepts, theta_dim, num_channel = num_channels)
    aggregator   = StudentAggregator(concept_dim, num_classes)
    student_concept_kd_model= StudentGSENN (conceptizer, parametrizer, aggregator)
    
    trainer = ClassificationTrainer(student_concept_kd_model, num_classes)
    trainer.train(train_loader, val_loader = valid_loader, teacher_model = teacher_model, epochs = epochs, save_path = None,with_concepts=True)
    trainer.plot_losses(save_path = '/home/cs20btech11042/XML_KD/experiments/student_concept_kd')
    student_concept_kd_model.eval()
    student_concept_kd_accuracy = trainer.evaluate(test_loader)
    
    model_dict = {
        'teacher':teacher_model,
        'student':student_model,
        'student_kd':student_kd_model,
        'student_concept_kd':student_concept_kd_model
    }
        
    generate_explantions(model_dict, test_loader, num_concepts)
    
    print(f'Accuracy of the Teacher Model is: {teacher_accuracy}')
    print(f'Accuracy of the Student Model is: {student_accuracy}')
    print(f'Accuracy of the Student Model with KD on the outputs is: {student_kd_accuracy}')
    print(f'Accuracy of the Student Model with concept KD is: {student_concept_kd_accuracy}')
    

if __name__ == '__main__':
    main()