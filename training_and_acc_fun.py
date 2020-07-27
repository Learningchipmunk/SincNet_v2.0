import tqdm
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from utils import plot_grad_flow, mixup

## Local files imports:
from Confusion_Matrix import confusion_matrix

## <!>---------------------------- Training and accuracy functions ----------------------------<!>
def accuracy(net, test_loader, criterion, n_classes,
             ## SincNet Params
             Batch_dev, wlen, wshift, 
             ## Confusion_Matrix param:
             matrix_name, compute_matrix = False,
             cuda=True):
    """ This is the function that computes the accuracy of the Network.

    Args:
        net (nn.Module): inherits from nn.Module and is a network.
        test_loader (torch.utils.data.DataLoader): the test data loader.
        criterion (nn.Loss): is the loss function.
        Batch_dev (int): is the number of test tensors that are stored at once in test loop.
        wlen (int): is the size of the input of the network, for SincNet it is considered the window size of the audio.
        wshift (int): is the step in between each window.
        matrix_name (str): is the name of the confusion matrix that will be saved.
        compute_matrix (bool, optional): Indicates if user wants to compute and save confusion matrix. Defaults to False.
        cuda (bool, optional): Indicates if net is on cuda. Defaults to True.

    Returns:
        (float, float, float): (best class error, validation loss, mean error on each window)
    """


    ## Modifs pour SincNet
    net.eval()
    

    loss_sum=0
    err_sum=0
    err_sum_snt=0
    ## End
    
    with torch.no_grad():
        ## Number of batches in test_loader
        snt_te = 0

        # Initializes confusion matrix and quantity to None:
        mat = None
        qty = None

        ## Initialisation of storing units for testing:
        stored_outs    = 0
        stored_labels  = 0
        stored_idx     = 0


        ## For SincNet, information in testloader is raw and can be of various lengths!
        for data in test_loader:
            ## Stores data from test_loader:
            audios, labels, file_ids = data
            
            ## Switches to cuda:
            if cuda:
                audios = audios.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.LongTensor)
            
            
            ## <!> Splits the data by file: time_complexity = O(batch_size) and space_complexity = O(Number_Of_Files)
            section  = []
            ids_list = []
            current_idx = file_ids[0].item()
            ids_list.append(current_idx)
            for i, el in enumerate(file_ids):
                current_el = el.item()
                if current_el != current_idx:
                    current_idx = current_el
                    ids_list.append(current_idx)
                    section.append(i - sum(section))

            ## Last section is added:    
            section.append(audios.size(0) - sum(section))

            ## Algorithm that splits the tensors by file:
            X_split = torch.split(audios, section)
            y_split = torch.split(labels, section)

            assert(len(X_split) == len(y_split))
            ## <!> EndSplit <!>

            
            ## Processes file by file the data:
            for i in range(len(X_split)):# nbre_fichier_par_batch= len(X_split)
                ## Stores the current id:
                current_idx    = ids_list[i]
                current_data   = X_split[i] if len(X_split) >= 1 else X_split
                current_labels = y_split[i] if len(y_split) >= 1 else y_split

                if type(stored_outs) is not torch.Tensor:
                    stored_idx    = current_idx
                    stored_outs   = net(current_data)
                    stored_labels = current_labels

                else:

                    if(stored_idx == current_idx):
                        stored_outs   = torch.cat([stored_outs, net(current_data)], dim=0)
                        stored_labels = torch.cat([stored_labels, current_labels], dim=0)

                    else:
                        #here, pout is equal to stored_outs

                        ## Predicts for every chunk of audio the label and counts how many time it got it correctly
                        pred = torch.max(stored_outs,dim=1)[1]
                        loss = criterion(stored_outs, stored_labels.long())
                        err  = torch.mean((pred!=stored_labels.long()).float())

                        ## Updates the confusion matrix:
                        if(compute_matrix):
                            mat, qty = confusion_matrix(mat, qty, pred=pred, labels=stored_labels, write_results = False, name = "Pas Important", cuda = True)

                        ## Updates the error that I use here:
                        loss_sum=loss_sum+loss.detach()
                        err_sum=err_sum+err.detach()

                        ## Sum the probability over the columns, then it stores the value and the position of the max. (Lionel's Method)
                        [val,best_class]=torch.max(torch.sum(stored_outs,dim=0), 0)
                        err_sum_snt=err_sum_snt+(best_class!=stored_labels[0]).float()

                        ## Stores new data:
                        stored_outs   = net(current_data)
                        stored_labels = current_labels
                        stored_idx    = current_idx

                        ## Updates the number of signals
                        snt_te  += 1
        
        ## Last File is not processed by the algorithm above...
        if(stored_outs.size(0) != 0):
            ## Predicts for every chunk of audio the label and counts how many time it got it correctly
            pred = torch.max(stored_outs,dim=1)[1]
            loss = criterion(stored_outs, stored_labels.long())
            err  = torch.mean((pred!=stored_labels.long()).float())

            ## Updates the confusion matrix:
            if(compute_matrix):
                mat, qty = confusion_matrix(mat, qty, pred=pred, labels=stored_labels, write_results = False, name = "Pas Important", cuda = True)

            ## Updates the error that I use here:
            loss_sum=loss_sum+loss.detach()
            err_sum=err_sum+err.detach()

            ## Sum the probability over the columns, then it stores the value and the position of the max. (Lionel's Method)
            [val,best_class]=torch.max(torch.sum(stored_outs,dim=0), 0)
            err_sum_snt=err_sum_snt+(best_class!=stored_labels[0]).float()
            
            ## Updates the number of signals
            snt_te  += 1

        ## Deletes everything that is stored:
        del stored_outs, stored_labels                  

        ## mean Error of best class:
        err_tot_dev_snt=err_sum_snt/snt_te
        
        ## mean Loss:
        loss_tot_dev=loss_sum/snt_te
        
        ## mean Error on each window:
        err_tot_dev=err_sum/snt_te
        
        ## Plots and saves the confusion matrix:
        if(compute_matrix):
            mat = confusion_matrix(mat, qty, write_results = True, name = matrix_name, cuda = True)
            
            
    net.train()

    return (err_tot_dev_snt, loss_tot_dev, err_tot_dev)



def train(net, optimizer, train_loader, valid_loader, criterion, criterion_onehot,
          ## Data scpecific variables:
          wlen,
          wshift,
          n_classes,
          ## File variables:
          output_folder,
          fname,
          Models_file_extension,
          ## Hyper param:
          n_epoch = 5,
          patience = 4,
          Batch_dev = 32,#Number of batches for testing set
          train_acc_period = 100,
          test_acc_period = 5,
          ## For mixup:
          beta_coef = 0.5,
          mixup_batch_prop = 0.5,
          use_mixup = False,
          same_classes = False,
          ## If a Net was loaded:
          starting_epoch = 0,
          ## If user wishes to plot grad:
          plotGrad = False,
          ## If user wishes to use a scheduler:
          use_scheduler = False,
          scheduler = None,
          ## If user wishes to save and compute confusion matrix:
          compute_matrix = False,
          ## Indicates if the network that is trained is SincNet
          is_SincNet = False,
          ## Is Cuda activated?
          cuda=True):
    """The train function

    Args:
        net (nn.Module): inherits from nn.Module and is a network.
        optimizer (torch.optim): Net optimizer for learning.
        train_loader (torch.utils.data.DataLoader): the train data loader.
        valid_loader (torch.utils.data.DataLoader): the test data loader.
        criterion (nn.Loss): is the loss function.
        criterion_onehot (nn.Loss): is the loss function compatible with one hot encoding.
        wlen (int): is the size of the input of the network, for SincNet it is considered the window size of the audio.
        wshift (int): is the step in between each window.
        n_classes (int): is the total number of classes of the dataset.
        output_folder (String): Name of the folder where we are going to write info.
        fname (String): The name of the model, will be used to name the saved files.
        Models_file_extension (String): The file extension of the saved models
        n_epoch (int, optional): The desired number of epoch, can be overwritten by early stopping. Defaults to 5.
        patience (int, optional): The patience of the early stopping algorithm. Defaults to 4.
        Batch_dev (int, optional): is the number of test tensors that are stored at once in the test loop. Defaults to 32.
        test_acc_period (int, optional): The period, in epoch, of each validation test. Defaults to 5.
        beta_coef (float, optional): If using mixup, this is the parameter of the Beta(beta_coef, beta_coef) distribution. Defaults to 0.5.
        use_mixup (bool, optional): Indicates if the user desires to do mixup. Defaults to False.
        sameClasses (bool, optional): If the user chooses to apply mixup only on data with same classes,
                                he needs to set this attribute's value to True. Defaults to False.
        starting_epoch (int, optional): The initial starting epoch. (If the model was loaded, it can be different from 0). Defaults to 0.
        plotGrad (bool, optional): Indicates if the user desires to plot Gradient flow. Defaults to False.
        use_scheduler (bool, optional): Indicates if the user desires to use a scheduler, it needs to be set to True if so. Defaults to False.
        scheduler (object, optional) : Is the scheduler for thje network. Defaults to None.
        compute_matrix (bool, optional): Indicates if user wants to compute and save confusion matrix. Defaults to False.
        is_SincNet (bool, optional): Indicates if the network that is trained is SincNet, it should be set to True if so. Defaults to False.
        cuda (bool, optional): Indicates if net is on cuda. Defaults to True.
    """


    net.train()

    ## Initialization:
    min_loss = float("inf")
    
    ## best_epoch_number is a var that stores the number of epoch required for best performance:
    best_epoch_number = 0
    
    ## p is a counter for how many accuracy checks we made without any improvement on validation loss:
    p=0
    
    ## Declaring to the user that training has begun
    print("Trainining begun with a patience of {} accuracy periods".format(patience), end="")
    if(use_mixup):
        string = ""
        if(same_classes):
            string = "Same Class "
        print(" and using {1}mixup with a Beta({0}, {0}) distribution and a mixup proportion = {2}.".format(beta_coef, string, mixup_batch_prop))
    else:
        print(".")
    if use_scheduler:
        print("Training is optimized with a scheduler.")
    print("Total number of classes is equal to : {}".format(n_classes))
    
    ## Continues training beyond n_epoch if algorithm did not converge:
    while(p < patience):
        
        for epoch in tqdm.tqdm(range(starting_epoch + 1, n_epoch + starting_epoch + 1)):  # loop over the dataset multiple times

            ## Stops the training if we exceeded its patience!
            if(p >= patience):
                break


            running_loss = 0.0
            running_acc = 0.0
            running_mixup_percentage = 0.0

            for i, data in enumerate(train_loader, 0):
                
                # Getting Mixed up data if required by user:
                if(use_mixup):
                    inputs, labels, mixup_states     = data
                    mixup_percentage                 = mixup_states.sum().item() / inputs.size(0)
                    running_mixup_percentage         = 0.33*mixup_percentage + 0.66*running_mixup_percentage
                else:
                    # gets the regular inputs
                    inputs, labels = data


                if cuda:
                    inputs = inputs.type(torch.cuda.FloatTensor)
                    labels = labels.type(torch.cuda.LongTensor)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)

                ## Loss evaluation:
                    # We use custom made function if there is mixup envolved
                    # Else, we use regular criterion from pytorch
                if(use_mixup):
                    loss = criterion_onehot(outputs, labels.long())
                else:
                    loss = criterion(outputs, labels.long())
                    
                    
                loss.backward()

                ## Plotting the grad for frequencies and second layer 1Dconv:
                if(plotGrad):
                    plot_grad_flow(net.named_parameters())   
                    #plot_grad_flow_simple(CNN_net.named_parameters())

                optimizer.step()
    
                ## If we used mixup, we need to convert back labels to og format.
                if(use_mixup):
                    labels = torch.max(labels, dim = 1)[1]

                running_loss = 0.33*loss.detach() + 0.66*running_loss
                predicted = torch.max(outputs.data, dim = 1)[1]

                ## @correct is the percentage of correct answers by the Net.
                correct = (predicted == labels).sum().item()/labels.size(0)
                running_acc = 0.33*correct + 0.66*running_acc

                # prints statistics during epoch!
                if i % train_acc_period == train_acc_period-1:
                    print("Training set : ")
                    print('[%d, %5d] running loss: %.3f' %(epoch, i + 1, running_loss))
                    print('[%d, %5d] running acc: %.3f' %(epoch, i + 1, running_acc))
                    
                    if use_mixup:
                        print('[%d, %5d] running mixup percentage: %.3f' %(epoch, i + 1, running_mixup_percentage))



            ## Validation loop part:
            if epoch % test_acc_period == 0:
                best_class_error, cur_loss, window_error = accuracy(net, valid_loader, criterion, n_classes,
                                                                    Batch_dev, wlen, wshift,
                                                                    matrix_name = fname, compute_matrix = compute_matrix,
                                                                    cuda=cuda)

                ## If user wishes to use a scheduler:
                if(use_scheduler):
                    scheduler.step(cur_loss)


                ## Writing the results in the specified file:
                with open(output_folder+"/" + fname + ".res", "a") as res_file:
                    res_file.write("epoch %i, running_loss_tr=%f running_acc_tr=%f best_class_acc_te=%f loss_te=%f window_acc_te=%f \n" % 
                                                                                                 (epoch,
                                                                                                 running_loss,
                                                                                                 running_acc,
                                                                                                 1-best_class_error,
                                                                                                 cur_loss,
                                                                                                 1-window_error))   

                print("\n")
                print("Validation set : ")
                print('[%d] test loss: %.3f'       %(epoch, cur_loss))
                print('[%d] window acc: %.3f'      %(epoch, 1-window_error))
                print('[%d] best class acc: %.3f'  %(epoch, 1-best_class_error))


                if(cur_loss < min_loss):
                    ## Saves the new loss:
                    min_loss = cur_loss

                    ## Saves the parameters if they are better:
                    # SincNet saving method:
                    if is_SincNet:
                        torch.save(net.CNN_net.state_dict(), output_folder + '/' + fname + "_CNN" + Models_file_extension)
                        torch.save(net.DNN1_net.state_dict(), output_folder+ '/' + fname + "_DNN1" + Models_file_extension)
                        torch.save(net.DNN2_net.state_dict(), output_folder+ '/' + fname + "_DNN2" + Models_file_extension)
                    
                    # Regular save:
                    torch.save(net.state_dict(), output_folder+ '/' + fname + "_Main_net" + Models_file_extension)
                        

                    ## Resets the patience, we found a better net.
                    p = 0

                    ## Stores the best number of epoch:
                    best_epoch_number = epoch

                else:
                    p +=1
        
        ## Inside While scope:
        starting_epoch += n_epoch#Here we go again...


      
    print('Finished Training, the best number of epoch was {}.'.format(best_epoch_number))
    
    # If user wants to plot grad: 
    if(plotGrad):
        ## Saves figure:
        plt.savefig("Images/" + fname + "_GradFlow.png", format = 'png')
        
        ## Then shows the figure:
        plt.show()
        
