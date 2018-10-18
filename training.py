"""
training for CNN denoising

Copyright (C) 2018-2019, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
"""

def check_accuracy(model, loss_fn, dataloader):
    """
    auxiliary function that computes mean of the loss_fn 
    over the dataset given by dataloader
    """
    import torch

    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()
        dtype   = torch.cuda.FloatTensor
        
    loss = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for t, (x, y) in enumerate(dataloader):
            x_var = x.type(dtype)
            y_var = y.type(dtype)
            out   = model(x_var)
            loss += loss_fn(out, y_var)
        return loss/(t+1)
        
        

def trainmodel(model, loss_fn, loader_train, loader_val=None, 
               optimizer=None, scheduler=None, num_epochs = 1, 
               save_every=10, loss_every=10, filename=None):
    """
    function that trains a network model
    Args:
        - model       : network to be trained
        - loss_fn     : loss functions
        - loader_train: dataloader for the training set
        - loader_val  : dataloader for the validation set (default None)
        - optimizer   : the gradient descent method (default None)
        - scheduler   : handles the hyperparameters of the optimizer
        - num_epoch   : number of training epochs
        - save_every  : save the model every n epochs
        - filename    : base filename for the saved models
        - loss_every  : print the loss every n epochs
    Returns:
        - model          : the trained network 
        - loss_history   : the history of loss values on the training set
        - valloss_history: the history of loss values on the validation set 
    """ 
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from time import time
    import numpy as np    
    
    dtype = torch.FloatTensor
    # GPU
    if torch.cuda.is_available():
        model   = model.cuda()
        loss_fn = loss_fn.cuda()
        dtype   = torch.cuda.FloatTensor
   

    if optimizer == None or scheduler == None:
        # the optimizer is in charge of updating the parameters of the model
        # it has hyperparameters for controlling the learning rate (lr) and 
        # the regularity such as the weight_decay 
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), 
                               eps=1e-08, weight_decay=0.00001, amsgrad=False)

        # the learning rate scheduler monitors the evolution of the loss
        # and adapts the learning rate to avoid plateaus 
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                            factor=0.9, patience=5000, verbose=True, 
                            threshold=0.0001, threshold_mode='rel', 
                            cooldown=0, min_lr=0, eps=1e-08)  
    
    loss_history=[]
    valloss_history=[]
    
    for epoch in range(num_epochs):
        
        for t, (x, y) in enumerate(loader_train):
            # make sure that the models is in train mode
            model.train()  

            # Apply forward model and compute loss on the batch
            x_var = x.type(dtype)
            y_var = y.type(dtype)
            out = model(x_var)  
            loss = loss_fn(out, y_var)
            
            # Zero out the gradients of parameters that the optimizer 
            # will update. The optimizer is already linked to the 
            optimizer.zero_grad()

            # Backwards pass: compute the gradient of the loss with
            # respect to all the learnable parameters of the model.
            loss.backward()

            # Update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            # Display current loss and compute validation loss
            if t % loss_every == 0:
                valstring=''
                if loader_val is not None:
                    valloss   = check_accuracy(model, loss_fn, loader_val)
                    valstring = ', val_loss = %.4f'%valloss.item()
                    valloss_history.append(valloss)
                loss_history.append(loss)
                print('Epoch %4d/%4d, ' % (epoch + 1, num_epochs) +  
                      'Iteration %d, loss = %.4f%s'% (t, loss.item(), valstring))
                
        if filename and ((epoch+1) % save_every == 0):
                torch.save([model, optimizer, loss_history, valloss_history], 
                           filename+'%s.pt' %int(epoch+1))

        # scheduler update
        scheduler.step(loss.data)
              
    return model, loss_history, valloss_history