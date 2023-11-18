import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import shutil
import time
import matplotlib.pyplot as plt

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def save_checkpoint(state, is_best, outpath):
    if outpath == None:
        outpath = os.path.join('../', 'checkpoints')

    #outdir = os.path.join(outpath, '{}_LR{}_Lambda{}'.format(state['theta_reg_type'],state['lr'],state['theta_reg_lambda']))
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filename = os.path.join(outpath, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(outpath,'model_best.pth.tar'))


class ClassificationTrainer():
    def __init__(self, model, num_classes):
        self.model = model
        self.cuda = torch.cuda.is_available()
        self.nclasses = num_classes
        self.prediction_criterion = F.cross_entropy

        self.learning_h = True
        self.h_sparsity = 1e-4
        self.alpha = 0.8
        self.T = 5
        self.h_reconst_criterion = F.mse_loss

        self.loss_history = []  # Will acumulate losse
        self.print_freq = 10 # need to check

        optim_betas = (0.9, 0.999)
        self.optimizer = optim.Adam(self.model.parameters(), lr= 1e-4, betas=optim_betas)

        if self.cuda:
            self.model = self.model.cuda()

    def train(self, train_loader, teacher_model = None, val_loader = None, with_concepts = False, epochs = 10,  save_path = None):
        best_prec1 = 0
        for epoch in range(epochs):
            self.train_epoch(epoch, train_loader,teacher_model,with_concepts)
            if val_loader is not None:
                val_prec1 = self.validate(val_loader) # Ccompytes acc

            is_best = val_prec1 > best_prec1
            best_prec1 = max(val_prec1, best_prec1)
            if save_path is not None:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'lr': self.args.lr,
                    'theta_reg_lambda': self.args.theta_reg_lambda,
                    'theta_reg_type': self.args.theta_reg_type,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : self.optimizer.state_dict(),
                    'model': self.model  
                 }, is_best, save_path)

        print('Training done')

    def train_batch(self, inputs, targets):
        self.optimizer.zero_grad()
        
        output = self.model(inputs)
        pred = F.log_softmax(output, dim = 1)

        pred_loss = self.prediction_criterion(pred, targets)
        all_losses = {'prediction': pred_loss.item()}
        
        h_loss = self.concept_learning_loss(inputs, all_losses)
        loss = pred_loss + h_loss

        loss.backward()
        self.optimizer.step()

        return pred, loss, all_losses
    
    def train_batch_kd(self, inputs, targets, teacher_model, with_concepts = False):
        self.optimizer.zero_grad()
        
        with torch.no_grad():
            teacher_output = teacher_model(inputs)

        output = self.model(inputs)
        pred = F.log_softmax(output, dim = 1)
        
        pred_loss = nn.KLDivLoss()(F.log_softmax(output/self.T, dim=1),
                             F.softmax(teacher_output/self.T, dim=1)) * (self.alpha * self.T * self.T) + \
              self.prediction_criterion(pred, targets) * (1. - self.alpha)
    
        all_losses = {'prediction': pred_loss.item()}
        
        if with_concepts:
            with torch.no_grad():
                teacher_encoding = teacher_model.conceptizer(inputs)[0]
            student_encoding = self.model.conceptizer(inputs)[0]
            h_loss = F.mse_loss(teacher_encoding,student_encoding) * self.alpha + \
                self.concept_learning_loss(inputs, all_losses) * (1. - self.alpha)
        else:
            h_loss = self.concept_learning_loss(inputs, all_losses)
        
        loss = pred_loss + h_loss   

        loss.backward()
        self.optimizer.step()

        return pred, loss, all_losses

    def concept_learning_loss(self, inputs, all_losses):
        recons_loss = self.h_reconst_criterion(
            self.model.recons, inputs.detach().requires_grad_(False))

        all_losses['reconstruction'] = recons_loss.item()
        
        sparsity_loss   = self.model.h_norm_l1.mul(self.h_sparsity)
        all_losses['h_sparsity'] = sparsity_loss.item()
        recons_loss += sparsity_loss
        return recons_loss

    def train_epoch(self, epoch, train_loader,teacher_model,with_concepts):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.train()
        end = time.time()

        for i, (inputs, targets) in enumerate(train_loader, 0):
            # measure data loading time
            data_time.update(time.time() - end)

            # get the inputs
            if self.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets = inputs.float().requires_grad_(), targets

            if teacher_model:
                outputs, loss, loss_dict = self.train_batch_kd(inputs, targets,teacher_model,with_concepts)
            else:
                outputs, loss, loss_dict = self.train_batch(inputs, targets)
            loss_dict['iter'] = i + (len(train_loader)*epoch)
            self.loss_history.append(loss_dict)

            prec1, prec5 = self.accuracy(outputs.data, targets.data, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]  '
                      'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                      #'Data {data_time.val:.2f} ({data_time.avg:.2f})  '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))


    def validate(self, val_loader, fold = None):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()
        end = time.time()
        
        for i, (inputs, targets) in enumerate(val_loader):
            # get the inputs
            if self.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                input_var = inputs
                target_var = targets

                # compute output
                output = self.model(input_var)
                loss   = self.prediction_criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = self.accuracy(output.data, targets, topk=(1, 5))

                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1[0], inputs.size(0))
                top5.update(prec5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        return top1.avg

    def evaluate(self, test_loader, fold = None):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, targets in test_loader:
            if self.cuda:
                data, targets = data.cuda(), targets.cuda()

            with torch.no_grad():
                output = self.model(data)
                test_loss += self.prediction_criterion(output, targets).item()

                pred = output.data.max(1)[1] # get the index of the max log-probability
                correct += pred.eq(targets.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(test_loader)
        fold = '' if (fold is None) else ' (' + fold + ')'
        acc = 100. * correct / len(test_loader.dataset)
        print('\nEvaluation{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            fold, test_loss, correct, len(test_loader.dataset),acc))
        return acc

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def plot_losses(self, save_path=None):
        loss_types = [k for k in self.loss_history[0].keys() if k != 'iter']
        losses = {k: [] for k in loss_types}
        iters = []
        for e in self.loss_history:
            iters.append(e['iter'])
            for k in loss_types:
                losses[k].append(e[k])
        fig, ax = plt.subplots(1, len(loss_types), figsize=(4 * len(loss_types), 5))
        if len(loss_types) == 1:
            ax = [ax]  # Hacky, fix
        for i, k in enumerate(loss_types):
            ax[i].plot(iters, losses[k])
            ax[i].set_title('Loss: {}'.format(k))
            ax[i].set_xlabel('Iters')
            ax[i].set_ylabel('Loss')
        if save_path is not None:
            plt.savefig(save_path + '/training_losses.pdf', bbox_inches='tight', format='pdf', dpi=300)
