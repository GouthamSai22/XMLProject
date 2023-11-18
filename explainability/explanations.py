import matplotlib.pyplot as plt
from os import path
import numpy as np
import torch
from torchvision.utils import make_grid

plt.style.use('seaborn-paper')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def create_barplot(ax, relevances, y_pred, x_lim=1.1, title='', x_label='', concept_names=None):
    # Example data
    y_pred = y_pred.item()
    if len(relevances.squeeze().size()) == 2:
        relevances = relevances[:, y_pred]
    relevances = relevances.squeeze()
    if concept_names is None:
        concept_names = ['C. {}'.format(i + 1) for i in range(len(relevances))]
    else:
        concept_names = concept_names.copy()
    concept_names.reverse()
    y_pos = np.arange(len(concept_names))
    colors = ['b' if r > 0 else 'r' for r in relevances]
    colors.reverse()

    ax.barh(y_pos, np.flip(relevances.detach().cpu().numpy()), align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(concept_names)
    ax.set_xlim(-x_lim, x_lim)
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_title(title, fontsize=18)

def save_or_show(img, save_path):
    # TODO: redesign me
    img = img.clone().squeeze()
    npimg = img.cpu().numpy()
    if len(npimg.shape) == 2:
        if save_path is None:
            plt.imshow(npimg)
            plt.show()
        else:
            plt.imsave(save_path, npimg)
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
    plt.clf()

def show_explainations(model_dict, test_loader, num_explanations = 10, save_path = None, batch_size = 128):
    # select test example
    iterator = iter(test_loader)
    (test_batch, test_labels) = next(iterator)
    batch_idx = np.random.randint(0, batch_size - 1, num_explanations)
    
    for name, model in model_dict.items():
        new_save_path = path.join(save_path, name)
        
        device = 'cuda:0' if next(model.parameters()).is_cuda else 'cpu'
        model.eval()
        test_batch = test_batch.float().to(device)
        
        with torch.no_grad():
            y_pred = model(test_batch)
            concepts, _ = model.conceptizer(test_batch)
            relevances = model.parametrizer(test_batch)
            if len(y_pred.size()) > 1:
                y_pred = y_pred.argmax(1)

        concepts_min = concepts.min().item()
        concepts_max = concepts.max().item()
        concept_lim = abs(concepts_min) if abs(concepts_min) > abs(concepts_max) else abs(concepts_max)

        plt.style.use('seaborn-paper')
        for i in range(num_explanations):
            gridsize = (1, 3)
            fig = plt.figure(figsize=(9, 3))
            ax1 = plt.subplot2grid(gridsize, (0, 0))
            ax2 = plt.subplot2grid(gridsize, (0, 1))
            ax3 = plt.subplot2grid(gridsize, (0, 2))

            # figure of example image
            image = test_batch[batch_idx[i]].permute(1, 2, 0).cpu()
            ax1.imshow((image-image.min())/(image.max()-image.min()))
            ax1.set_axis_off()
            #ax1.set_title(f'Input Prediction: {class_names[y_pred[batch_idx[i]].item()]}', fontsize=18)
            ax1.set_title(f'Input Prediction: {y_pred[batch_idx[i]].item()}', fontsize=18)

            create_barplot(ax2, relevances[batch_idx[i]], y_pred[batch_idx[i]], x_label='Relevances (theta)')
            ax2.xaxis.set_label_position('top')
            ax2.tick_params(which='major', labelsize=12)

            create_barplot(ax3, concepts[batch_idx[i]], y_pred[batch_idx[i]], x_lim=concept_lim,
                        x_label='Concept activations (h)')
            ax3.xaxis.set_label_position('top')
            ax3.tick_params(which='major', labelsize=12)

            plt.tight_layout()

            plt.show() if save_path is None else plt.savefig(path.join(new_save_path,'explanation_{}.png'.format(i)))
            plt.close('all')


def show_prototypes(model, test_loader, num_concepts, num_prototypes=10, save_path=None):
    model.eval()
    activations = []
    for x, _ in test_loader:
        x = x.float().to("cuda:0" if next(model.parameters()).is_cuda else "cpu")
        with torch.no_grad():
            concepts, _ = model.conceptizer(x)
            activations.append(concepts.squeeze())
    activations = torch.cat(activations)

    _, top_test_idx = torch.topk(activations, num_prototypes, 0)

    top_examples = [test_loader.dataset.data[top_test_idx[:, concept].cpu().numpy()] for concept in range(num_concepts)]
    # flatten list and ensure correct image shape
    top_examples = [img.unsqueeze(0) if len(img.shape) == 2 else torch.from_numpy(np.moveaxis(img, -1, 0)) for sublist in top_examples for img in sublist]


    plt.rcdefaults()
    fig, ax = plt.subplots()
    concept_names = ['Concept {}'.format(i + 1) for i in range(num_concepts)]

    start = 0.0
    end = num_concepts * x.size(-1)
    stepsize = abs(end - start) / num_concepts
    ax.yaxis.set_ticks(np.arange(start + 0.5 * stepsize, end - 0.49 * stepsize, stepsize))
    ax.set_yticklabels(concept_names)
    plt.xticks([])
    ax.set_xlabel('{} most prototypical data examples per concept'.format(num_prototypes))
    ax.set_title('Concept Prototypes: ')
    save_or_show(make_grid(top_examples, nrow = num_prototypes, pad_value = 1), save_path)
    plt.rcdefaults()
    
def generate_explantions(model_dict, test_loader, num_concepts):
    show_explainations(model_dict, test_loader, save_path = f'XML_KD/experiments')
    for name, model in model_dict.items():
        show_prototypes(model, test_loader, num_concepts, save_path = f'XML_KD/experiments/{name}/prototypes')
        
    