from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

import torch
from src.simplex import Simplex
from src.dataset import load_adult, load_diabetes, load_hospital, load_mnist

            
from alibi.explainers import Counterfactual, CounterfactualProto
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def load_columns(dataset):
    if dataset == 'adult':
        cols = ['workclass', 'education', 'marital-status', 'occupation', \
       'relationship', 'race', 'gender', 'native-country', 'capital-gain', \
       'capital-loss', 'hours-per-week', 'age']
    elif dataset == 'diabetes':
        cols = ['GenHlth', 'Age', 'Education', 'Income', \
        'HighBP','BMI','HighChol','DiffWalk',\
        'HeartDiseaseorAttack','PhysHlth',\
        'HvyAlcoholConsump','Sex','CholCheck']
    elif dataset == 'hospital':
        cols = ['Gender', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', \
       'Alcoholism', 'SMS_received', 'Handcap', 'Age', 'ScheduleDays']
    else:
        assert False, f'dataset should be one of adult, diabetes, hospital. Got {dataset}.'
    return cols


def get_simplex(model, corpus, test, verbose = False):
    corpus_latents = model.latent_representation(corpus).detach()
    test_latents = model.latent_representation(test).detach()
    
    simplex = Simplex(corpus_examples=corpus,
                  corpus_latent_reps=corpus_latents)
    
    simplex.fit(test_examples=test, test_latent_reps=test_latents, verbose = verbose)
    return simplex

def transform_x(inp, scaler, encoder):
    return torch.Tensor(scaler.transform(encoder.transform(inp)))

def inverse_transform_x(inp, scaler, encoder, columns):
    inp = scaler.inverse_transform(inp.detach().numpy())
    inp = np.round(inp,0).astype(int)
    return encoder.inverse_transform(pd.DataFrame(inp, columns = columns))

def color_boolean(val):
    return f'color: {"red" if val else ""}'

def display_tabular_cfs(cfs_list, model, x, scaler, encoder, columns, filename, sources  = ['SimplexCF', 'NN', 'CFProto']):
    inverse_transform = partial(inverse_transform_x, scaler = scaler, encoder = encoder, columns = columns)
    records = []
    row = inverse_transform(x)
    row['label'] = model(x).argmax().item()
    row.set_index(pd.Index(['original']), inplace = True)
    records.append(row)

    for cfs, source in zip(cfs_list, sources):
        if cfs is not None:
            for i , kept_cf in enumerate(cfs):
                kept_cf = kept_cf.reshape(1,*kept_cf.shape)
                row = inverse_transform(kept_cf)
                row['label'] = model(kept_cf).argmax().item()
                row.set_index(pd.Index([f'{source}_counterfactual_{i+1}']), inplace = True)
                records.append(row)

    records = pd.concat(records).T
    
    records.style.apply(lambda _: records.apply(lambda x : x != records['original']).applymap(color_boolean), axis=None).to_excel(f'{filename}.xlsx')
    display(records.style.apply(lambda _: records.apply(lambda x : x != records['original']).applymap(color_boolean), axis=None))
            

def show_image(filename, x1,x2 = None,x3 = None, labels = None):
    if x2 is None:
        plt.figure(figsize = (1,2))
        plt.imshow(x1*0.3081 + 0.1307, cmap = 'gray')
        plt.show()
    else:
        plt.figure(figsize = (2,6))
        _, axarr = plt.subplots(1,3)
        axarr[0].set_title(f'Original,\nClass = {labels[0]}')
        axarr[1].set_title(f'Counterfactual,\nClass = {labels[1]}')
        axarr[2].set_title(f'Pixels Changed,\nDesired Class = {labels[2]}')

        axarr[0].imshow(x1*0.3081 + 0.1307, cmap = 'gray')
        axarr[1].imshow(x2*0.3081 + 0.1307, cmap = 'gray')
        axarr[2].imshow(x3*0.3081 + 0.1307, cmap = 'gray')
        plt.savefig(f'{filename}.png', dpi = 300)
        plt.show()

def display_image_cfs(cfs, model, x, desired_class, filename):
    if cfs is None:
        print('No counterfactuals found!')
    else:
        for kept_cf in cfs:
            kept_cf = kept_cf.reshape(1,*kept_cf.shape)
            #print('Original || Counterfactual || Diff ')
            # show_image(kept_cf[0][0])
            diff = x[0][0] != kept_cf[0][0]
            #print('Diff image: ')
            show_image(filename, x[0][0] , kept_cf[0][0] , diff, [model(x).argmax(), model(torch.Tensor(kept_cf)).argmax(), desired_class])
            #print('Predicted: ', model(torch.Tensor(kept_cf)).argmax(), ' || ', 'Desired: ', desired_class, ' || ', 'Orginal: ',  model(x).argmax())
            #print('Sparsity = ', (diff).float().mean())
            #print()


def get_cfproto_cf(X_corpus, model, x):
    shape = (1,) + X_corpus.shape[1:]
    predict_fn = lambda x: torch.exp(model(torch.Tensor(x))).detach().numpy()
    cf = CounterfactualProto(predict_fn, shape, use_kdtree = True)
    cf.fit(X_corpus, trustscore_kwargs=None)
    explanation = cf.explain(x)
    return torch.Tensor(explanation.cf['X'])

def get_cf_nproto_cf(X_corpus, model, x):
    shape = (1,) + X_corpus.shape[1:]
    target_proba = 1.0
    tol = 0.01 # want counterfactuals with p(class)>0.99
    target_class = 'other' # any class other than 7 will do
    max_iter = 100
    lam_init = 1e-1
    max_lam_steps = 10
    learning_rate_init = 0.1
    feature_range = (X_corpus.min(),X_corpus.max())

    predict_fn = lambda x: torch.exp(model(torch.Tensor(x))).detach().numpy()

    # initialize explainer
    cf = Counterfactual(predict_fn, shape=shape, target_proba=target_proba, tol=tol,
                        target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                        max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                        feature_range=feature_range)
    explanation = cf.explain(x)
    return torch.Tensor(explanation.cf['X'])

def get_simplex_cf_tabular(simplex, model, test_id, encoder, n_cfs = 3, mask = None,  use_attributions = True,
                    elastic_net_factor = 1, ae_factor = 1, target_factor = 1, prototype_factor = 1, map_categorical = True):
    x = simplex.test_examples[test_id : test_id+1]
    desired_class = model(x).topk(2).indices[0,1]
    
    cat_indices = list(range(len(encoder.cols)))

    cfs = simplex.get_counterfactuals(test_id = test_id, model = model, n_counterfactuals = n_cfs, mask = mask, map_categorical = map_categorical,
                                      cat_indices = cat_indices, elastic_net_factor =  elastic_net_factor, use_attributions = use_attributions,
                                      ae_factor =  ae_factor, target_factor = target_factor, prototype_factor = prototype_factor)
    return cfs, x, desired_class

def get_simplex_cf_image(simplex, model, test_id, n_cfs = 3, use_attributions = True, map_categorical = True,
                        elastic_net_factor = 1, ae_factor = 1, target_factor = 1, prototype_factor = 1):
    x = simplex.test_examples[test_id : test_id+1]
    desired_class = model(x).topk(2).indices[0,1]
    

    cfs = simplex.get_counterfactuals(test_id = test_id, model = model, use_attributions = use_attributions,
                                      n_counterfactuals = n_cfs, min_epochs = 50, \
                                    elastic_net_factor =  elastic_net_factor, map_categorical = map_categorical, 
                                      ae_factor =  ae_factor, target_factor = target_factor, prototype_factor = prototype_factor)
    return cfs, x, desired_class

def get_loader(dataset, k = 500):
    if dataset == 'adult':
        loader = iter(load_adult(k, train=False))
    elif dataset == 'diabetes':
        loader = iter(load_diabetes(k, train=False))
    elif dataset == 'hospital':
        loader = iter(load_hospital(k, train=False))
    elif dataset == 'mnist':
        loader = iter(load_mnist(k, train=False))
    else:
        loader = None
    return loader

def b_g(s, A, cmap='Greens', low=0, high=1):
    # Pass the columns from Dataframe A 
    a = A.loc[:,s.name].copy()
    rng = A.max().max() - A.min().min()
    norm = colors.Normalize(A.min().min() - (rng * low),
                        A.max().max() + (rng * high))
    normed = norm(a.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]