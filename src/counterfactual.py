from collections import Counter

from src.autoencoder import AutoEncoder
from src.prototype import prototype_selector

import hnswlib
import numpy as np
# from tqdm import tqdm

from sklearn.metrics.pairwise import euclidean_distances
import torch
import torch.nn.functional as F


def build_index(data, dim):
    p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip
    p.init_index(len(data))
    p.set_ef(10)
    p.set_num_threads(4)
    p.add_items(data, range(len(data)))
    return p

def nearest_neighbors_approximate(p, test, n_counterfactuals):
    labels, _ = p.knn_query(test, k=n_counterfactuals)
    topn = Counter(labels.flatten()).most_common(n_counterfactuals)
    return [int(i) for i,j in topn]

def nearest_neighbors_exact(data, test, n_counterfactuals):
    return list(np.argsort(euclidean_distances(test, data).mean(axis=0))[:n_counterfactuals].flatten())

def compute_ae_loss(data, latent_output, cat_indices):
    cat_mask = (torch.ones_like(data) == 0).float()
    cat_mask[:,cat_indices] = 1
    
    cont_loss =  F.mse_loss((1 - cat_mask) * data, (1 - cat_mask) * latent_output)
    cat_loss =   torch.eq(cat_mask * data, cat_mask * latent_output).sum()
    return cont_loss + cat_loss

def build_counterfactual_decoder(model, test, test_latent_approx, prototype, target, 
                                 cat_indices, device, task, lag, \
                                 epsilon_weight, l1_weight,\
                                 min_epochs, max_epochs, mask,\
                                elastic_net_factor, ae_factor, \
                                target_factor, prototype_factor): 
    
    size = np.prod(test.size())
    
    if task == 'regression':
        sign, target = target.split(',')
    
    autoencoder = AutoEncoder(model.latent_dim, size, mask)
    autoencoder.to(device)
    optimizer = torch.optim.Adam([param for param in autoencoder.parameters() if param.requires_grad == True])
    
    for i in range(max_epochs):
        data = test.to(device)
        optimizer.zero_grad()

        # Generate possible counterfactual x from latent space
        eps = torch.randn(1, model.latent_dim, requires_grad=True) * epsilon_weight
        latent_output= autoencoder(data,test_latent_approx,eps)
        latent_output = latent_output.reshape(size, -1)
        latent_output_reshaped = latent_output.reshape(*test.shape)

        # Compute the target loss 
        predicted = model.forward(latent_output_reshaped)
        desired , desired[0,target] = torch.ones_like(predicted) * -100 , 0
        
        if task == 'classification':
            desired , desired[0,target] = torch.ones_like(predicted) * -100 , 0 
        elif task == 'regression':
            desired, desired[0,1] = torch.ones((1,2)) * -100, 0
            if '>' in sign:
                bin_pred = predicted > float(target)
            else:
                bin_pred = predicted < float(target)
            if bin_pred is True:
                predicted = torch.ones((1,2)) * -100
                predicted[0,1] = 0
            elif bin_pred is False:
                predicted = torch.ones((1,2)) * -100
                predicted[0,0] = 0
            
        target_loss = F.kl_div(desired,predicted, reduction = 'batchmean',log_target = True)
        
        # Compute the autoencoder reconstruction loss
        if cat_indices is None:
            ae_loss = F.mse_loss(data, latent_output_reshaped)
        else:
            ae_loss = compute_ae_loss(data, latent_output, cat_indices)

        # Compute the prototype loss    
        encd_recon_x = model.latent_representation(latent_output_reshaped).reshape(model.latent_dim,1)
        prototype_loss = F.mse_loss(prototype, encd_recon_x)
        
        # Compute the elastic net regularization loss
        encd_x = model.latent_representation(data).reshape(model.latent_dim,1) 
        l1_loss = l1_weight * F.l1_loss(encd_x, encd_recon_x)
        l2_loss = F.mse_loss(encd_x, encd_recon_x)
        elastic_net_loss = l1_loss + l2_loss

        # Compute total loss
        total_loss = elastic_net_factor * elastic_net_loss + \
                    ae_factor * ae_loss + \
                    target_factor * target_loss + \
                    prototype_factor * prototype_loss 

        total_loss.backward()
        optimizer.step()

        # If counterfactual found, exit.
        ans = predicted.argmax()
        if ans == target and i >= min_epochs + lag:
            break
            
    return autoencoder


def get_attributions(inputs, baseline, model, n_bins):
    inputs = inputs.detach().clone().requires_grad_()
    inputs.retain_grad()

    # Hold the gradients for each step
    grads = []
    for k in range(1, n_bins + 1):
        model.zero_grad()
        # Interpolation from the baseline to the input
        baseline_input = baseline + ((k / n_bins) * (inputs - baseline))
        # Put the interpolated baseline through the model
        out = model(baseline_input)

        # Get the predicted classes and use them as indexes for which we want
        # attributions
        idx = out.argmax(dim=1).unsqueeze(1)
        # Select the output for each predicted class
        out = out.gather(dim=1, index=idx)

        # Perform backpropagation to generate gradients for the input
        out.backward(torch.ones_like(idx))

        # Append the gradient for each step
        grads.append(inputs.grad.detach())


    # Stack the list of gradients, compute the mean over the m steps
    grads = torch.stack(grads, 0).mean(dim=0)
    
    # Compute attributions
    attributions = (inputs - baseline).detach() * grads
    return attributions
    

def generate_candidates_with_neighbours(generated, neighbors, cat_indices, mask):
    # Compute candidates for categorical variables using latent neighbors of
    # the generated counterfactuals

    candidates = []
    mask = mask.flatten()
    cat_indices = [j for i, j in enumerate(cat_indices) if mask[i] == 1]
    
    for i, generated_ in enumerate(generated):
        generated_i = generated_.detach().clone()
        
        generated_i[cat_indices] = neighbors[i][cat_indices]
        candidates.append(torch.Tensor(generated_i.reshape(1,*generated_i.shape)))
    return torch.cat(candidates, dim = 0)

def get_baseline_counterfactuals(model, test, target, corpus, n_counterfactuals = 1):
    # Simple baseline counterfactual generator using proximity in latent space
    embs = model.latent_representation(test).detach().numpy()
    corpus =  corpus[model(corpus).argmax(axis=1) == target]
    corpus_ = model.latent_representation(corpus).detach().numpy()
    k_idx = nearest_neighbors_exact(embs,corpus_,n_counterfactuals)
    counterfactuals = corpus[k_idx]
    return counterfactuals

def get_counterfactuals(model, corpus, test, test_latent_approx, target,  \
                        mask = None, n_bins = 50, cat_indices = None, map_categorical = True, \
                    n_counterfactuals = 1, epsilon_weight = 1e-3, baseline = None, \
                        device = 'cpu', neighbors = 'exact', lag = 0, use_attributions = True, \
                    l1_weight = 1e-2, min_epochs = 50, max_epochs = 100, mins = None, maxs = None, \
                    elastic_net_factor = 1, ae_factor = 1, target_factor = 1, prototype_factor = 1):
    
    if model(test).shape[-1] > 1:
        task = 'classification'
    else:
        task = 'regression'
        
    if len(list(test.shape)) == 2:
        data_type = 'table'
    else:
        data_type = 'image'
    
    if data_type == 'image':
        minimum = torch.min(corpus)
        maximum = torch.max(corpus)
    else:
        if mins is None:
            minimum = torch.ones_like(test) * -1e255
        else:
            minimum = torch.Tensor(mins)

        if maxs is None:
            maximum = torch.ones_like(test) * 1e255
        else:
            maximum = torch.Tensor(maxs)
            
    if baseline is None:
        baseline = corpus.mean(axis = 0).requires_grad_()
        
    if mask is None:
        mask = torch.ones_like(test)
    else:
        mask = mask.reshape(*test.shape)
    
    mask = mask.reshape(-1,  np.prod(test.size()))

     # Using the protodash algorithm (https://arxiv.org/abs/1707.01212),
     # generate a prototype embedding for the target/desired class.   
    corpus =  corpus[model(corpus).argmax(axis=1) == target]
    corpus_ = model.latent_representation(corpus).detach().numpy()
    wt, idx, _ = prototype_selector(corpus_, corpus_, m = n_counterfactuals)
    prototype = torch.Tensor(np.dot(wt,corpus_[idx])).reshape(model.latent_dim,1)
    
    # Starting from simplex as the encoder, build an counterfactual autoencoder
    autoencoder = build_counterfactual_decoder(model = model, prototype = prototype, test = test, \
                                               test_latent_approx = test_latent_approx, \
                                    target = target, min_epochs = min_epochs, l1_weight = l1_weight,\
                                    max_epochs = max_epochs, epsilon_weight = epsilon_weight, lag = lag, \
                                    device = device, cat_indices = cat_indices, task = task, mask = mask, \
                                    elastic_net_factor = elastic_net_factor, ae_factor = ae_factor, \
                                    target_factor = target_factor, prototype_factor = prototype_factor)

    # Generate several possible counterfactual candidates from the autoencoder
    cfs = torch.cat([autoencoder(test,test_latent_approx,torch.randn(1, model.latent_dim) * epsilon_weight).reshape(*test.shape) \
                     for _ in range(n_counterfactuals)], axis = 0)
    counterfactuals = list()
    
    # If categorical values exist, 
    # modify candidates to use categories from neighbors in latent space.
    # Else, use candidates directly.
    if map_categorical is True:
        bool_idx = model(corpus).argmax(axis=1) == target
        sub_X = corpus[bool_idx]
        sub_X_ = corpus_[bool_idx]

        embs = model.latent_representation(cfs).detach().numpy()
        if neighbors == 'exact':
            k_idx = nearest_neighbors_exact(embs,sub_X_,n_counterfactuals)
        else:
            p = build_index(sub_X_, dim = model.latent_dim)
            k_idx = nearest_neighbors_approximate(p,embs,n_counterfactuals)

        shape = list(test.shape)
        shape[0] = n_counterfactuals
    
    if cat_indices is not None and map_categorical is True:
        inputs = sub_X[k_idx].reshape(shape)
        inputs = mask * inputs + (1 - mask) * test
        inputs_ = generate_candidates_with_neighbours(cfs.detach().clone(), 
                                                     inputs.detach().clone(), 
                                                     cat_indices, mask)

        inputs = torch.cat([inputs_,inputs])
        inputs = torch.unique(inputs, dim=0)
        inputs = inputs[model(inputs).argmax(axis = 1) == target]
        
    else:
        inputs = cfs.detach().clone()
        inputs = torch.unique(inputs, dim=0)
        inputs = inputs[model(inputs).argmax(axis = 1) == target]
    
    if use_attributions:
        # Get feature-level attributions for the candidates and use this to
        # additionally improve sparsity.
        attributions = get_attributions(inputs, baseline, model, n_bins)
        
        for i, attr in enumerate(attributions):
            attr = attr.reshape(1, *attr.shape)
            test_ = test.detach().clone()
            inp = inputs[i,:].detach().clone().reshape(*test.shape)
            kept_cf = None
            
            # Return a counterfactual such that the minimal possible change to original
            # instance required to change the target is returned.
            if data_type == 'table':
                f_attr = attr.squeeze().detach().numpy()
                if  f_attr.size > 1:
                    sorted_attrbs = sorted(f_attr)
                else:
                    sorted_attrbs = [f_attr]

                if len(sorted_attrbs) > 20:
                    vals_to_check = list(sorted(set([sorted_attrbs[int(i/100 * len(sorted_attrbs))] for i in range(0,100,5)])))
                else:
                    vals_to_check = [float(s) for s in sorted_attrbs]
            else:
                sorted_attrbs = sorted(attr[attr > 0].squeeze().detach().numpy())
                vals_to_check = [sorted_attrbs[int(i/100 * len(sorted_attrbs))] for i in range(0,100,2)]

            for threshold in reversed(vals_to_check):
                b = (attr >= threshold).float()
                test_ = (inp * b) + (test.detach().clone() * (1 - b))
                test_ = torch.clip(test_, minimum, maximum)
                pred = model(test_).argmax()

                if pred == target:
                    kept_cf = test_.detach().clone()
                    break

            if kept_cf is not None:
                counterfactuals.append(kept_cf)
    else:
        for cf in inputs:
            counterfactuals.append(cf)
    
    if len(counterfactuals) == 0:
        return
    
    counterfactuals = torch.unique(torch.vstack(counterfactuals),dim=0)
    
    # return top n_counterfactuals instances with the minimal sparsity
    if data_type == 'table':
        idx = (counterfactuals != test).sum(axis = 1)
        idx = idx.topk(min([n_counterfactuals, len(idx)]), largest = False, sorted = True)
        counterfactuals = counterfactuals[idx.indices]

    return counterfactuals 