from ops.misc import *

from time import time

import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader


class Trainer():
    def __init__(
        self, 
        dataset:Union[CustomLoader, Dataset], 
        model:torch.nn.Module,
        batch_size:int = 256,
        learning_rate:float = 1e-3,
        train_ratio:float = 0.7,
        test_ratio:float = 0.2,
        params:Generator = None,
    ) -> None:
        """
        Training module for a single dataset.

        Args:
            dataset (CustomLoader | Dataset) : dataset from which to load the data.
            model (torch.nn.Module) : model to be trained.
            batch_size (int, optional): how many samples per batch to load.
            learning_rate (float, optional): initial learning rate.
            train_ratio (float, optional) : % of training samples from whole dataset.
            test_ratio (float, optional) : % of test samples from whole dataset.
        """

        cudnn.benchmark = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # load data
        self.dloader = build_batches(dataset, batch_size, train_ratio, test_ratio)

        # build model
        self.md = model.to(self.device)
        self.cf = torch.nn.CrossEntropyLoss().to(self.device)

        if params == None :
            self.ot = optim.Adam(model.parameters(), lr=learning_rate)
        else :
            self.ot = optim.Adam(params, lr=learning_rate)
        
    def train(self, epochs:int = 10, iteration:int = 0) -> Tuple[List, List]:
        """
        Train a model.

        Args:
            epochs (int, optional): defines the number times that the full-batch learning will work.
            itertaion (int, optional): defines how much times minibatch should be taken.
                The training will be stopped after [%iteration] of minibatches are forwarded.
            
        Returns:
            metrics_train, metrics_eval (List): results of accuracy obtained at training/evaluation mode.
        """

        self.md.train()
        num_train = len(self.dloader['train'].dataset)
        num_minibatches = len(self.dloader['train'])
        
        metrics_train = []; metrics_eval = []

        stime = time()
        progress = 0
        record_delay = max(1, iteration//50)

        if iteration != 0 :
            epochs = iteration + 1
        
        for epoch in range(epochs):
            correct_all = 0
            loss_all = 0
            
            for i, (x_b, y_b) in enumerate(self.dloader['train']):
                progress += 1

                x_b, y_b = self.applycuda(x_b, y_b)

                # Forward
                self.ot.zero_grad()
                y_hat = self.md(x_b)
                
                loss = self.cf(y_hat, y_b)
                loss_all += loss.item()*len(y_b)

                # Inference
                pred = y_hat.argmax(dim=1, keepdim=True)
                correct = pred.eq(y_b.view_as(pred)).sum().item()
                correct_all += correct
                
                # Back-propagation
                loss.backward()
                self.ot.step()

                if iteration > 0 :
                    print("Training Progress : %d/%d"%(progress, iteration), end="\r")

                    if progress%record_delay == 0:
                        metrics_train.append({
                            'acc':correct/len(y_b),
                            'loss':loss
                        })

                        print("iter: {:5d}, train-loss: {:2.3f} train-acc: {:2.3f}".format(
                                progress, 
                                loss, 
                                correct/len(y_b)
                            ), end=' '
                        )

                        if self.dloader['eval'] != None :
                            e_acc, e_loss = self.eval()
                            metrics_eval.append({
                                'acc':e_acc,
                                'loss':e_loss
                            })

                        etime = time()
                        print("[in %.1f sec]"%(etime-stime))
                        stime = etime

                    if progress == iteration:
                        return metrics_train, metrics_eval
                else :
                    print("Training Progress : %d/%d"%(i+1, num_minibatches), end="\r")
                
            if iteration == 0 :
                print("epoch: {:3d}, train-loss: {:2.3f} train-acc: {:2.3f}".format(
                        epoch+1, 
                        loss_all/num_train, 
                        correct_all/num_train
                    ), end=' '
                )

                metrics_train.append({
                    'acc':correct_all/num_train,
                    'loss':loss_all/num_train
                })

                if self.dloader['eval'] != None :
                    e_acc, e_loss = self.eval()
                    metrics_eval.append({
                        'acc':e_acc,
                        'loss':e_loss
                    })

                etime = time()
                print("[in %.1f sec]"%(etime-stime))
                stime = etime
                
        return metrics_train, metrics_eval
    
    def eval(self):
        num_eval = len(self.dloader['eval'].dataset)
        self.md.eval()

        with torch.no_grad():
            correct_all = 0
            loss_all = 0
            
            for x_b, y_b in self.dloader['eval']:
                x_b, y_b = self.applycuda(x_b, y_b)

                # Forward/Inference
                y_hat = self.md(x_b)
                loss = self.cf(y_hat, y_b)
                loss_all += loss.item()*len(y_b)

                pred = y_hat.argmax(dim=1, keepdim=True)
                correct = pred.eq(y_b.view_as(pred)).sum().item()
                correct_all += correct

            print("eval-loss: {:2.3f} eval-acc: {:2.3f}".format(
                    loss_all/num_eval, 
                    correct_all/num_eval
                ), end=' '
            )

            self.md.train()

            return correct_all/num_eval, loss_all/num_eval

    def infer(self, x:Union[Dataset, DataLoader] = None, label:bool = True) -> Tuple[List, List]:
        """
        Inference the the hypothesis for the test samples.
        You can choose the default test dataloader("self.dloader['test']") or any other input tensor.

        Args:
            x (Dataset | DataLoader, Optional): target dataset.
            label (bool): if 'False', it means true labels are not given.

        Returns:
            y_real, y_pred (List): list of true labels and model's predictions.
        """

        self.md.eval()
        test_loss = 0
        correct = 0

        data_loader = x

        if issubclass(type(x), Dataset):
            data_loader = DataLoader(x, batch_size=128, shuffle=True)

        if data_loader == None :
            data_loader = self.dloader['test']

        num_test = len(data_loader.dataset)

        with torch.no_grad():
            y_real = []; y_pred = []

            if label :
                for x_b, y_b in data_loader:
                    x_b, y_b = self.applycuda(x_b, y_b)
                    
                    y_hat = self.md(x_b)
                    test_loss += self.cf(y_hat, y_b).item()*len(y_b)
                    pred = y_hat.argmax(dim=1, keepdim=True)
                    correct += pred.eq(y_b.view_as(pred)).sum().item()
                    
                    y_real += [i.cpu().numpy() for i in y_b]
                    y_pred += [i.cpu().numpy() for i in pred]
                    
                print('\nTest set: Average loss: {:.3f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
                        test_loss/num_test, 
                        correct, 
                        num_test, 
                        100. * correct / num_test
                    )
                )

                return np.array(y_real), np.array(y_pred)
            else :
                for x_b in data_loader:
                    x_b = self.applycuda(x_b)

                    y_hat = self.md(x_b)
                    pred = y_hat.argmax(dim=1, keepdim=True)
                    y_pred += [i.cpu() for i in pred]

                return None, np.array(y_pred)

    def applycuda(self, *args:Tuple[Any, ...]) -> List:
        """
        Transfers a tensor from CPU to GPU when the 'cuda' flag is on.

        Args:
            *args: list of original tensors.

        Returns:
            list of cuda tensors.
        """

        return [d.to(self.device) for d in args]