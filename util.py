from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from src.simplex import Simplex

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
    return f'background-color: {"green" if val else ""}'

def display_tabular_cfs(cfs, model, x, desired_class, scaler, encoder, columns):
    inverse_transform = partial(inverse_transform_x, scaler = scaler, encoder = encoder, columns = columns)
    transform = partial(transform_x, scaler = scaler, encoder = encoder)
    
    print('Original: ')
    display(inverse_transform(x))
    print()

    if cfs is None:
        print('No counterfactuals found!')
    else:
        for kept_cf in cfs:
            kept_cf = kept_cf.reshape(1,*kept_cf.shape)

            diff = (inverse_transform(x) != inverse_transform(kept_cf))
            #print(diff)
            #print()

            print('Kept counterfactual generation: ')
            display(inverse_transform(kept_cf).style.apply(lambda _: diff.applymap(color_boolean), axis=None))
            print()

            # print('Sparsity: ', int(diff.values.sum()))
            # print()

            print('Predicted: ', model(torch.Tensor(kept_cf)).argmax(), ' || ', 'Desired: ', desired_class, ' || ', 'Orginal: ',  model(x).argmax())

            print('*'*120)
            

def show_image(x1,x2 = None,x3 = None):
    if x2 is None:
        plt.figure(figsize = (1,2))
        plt.imshow(x1*0.3081 + 0.1307, cmap = 'gray')
        plt.show()
    else:
        plt.figure(figsize = (2,6))
        f, axarr = plt.subplots(1,3)
        axarr[0].imshow(x1*0.3081 + 0.1307, cmap = 'gray')
        axarr[1].imshow(x2*0.3081 + 0.1307, cmap = 'gray')
        axarr[2].imshow(x3*0.3081 + 0.1307, cmap = 'gray')
        plt.show()

def display_image_cfs(cfs, model, x, desired_class):
    if cfs is None:
        print('No counterfactuals found!')
    else:
        for kept_cf in cfs:
            kept_cf = kept_cf.reshape(1,*kept_cf.shape)
            print('Original || Counterfactual || Diff ')
            # show_image(kept_cf[0][0])
            diff = x[0][0] != kept_cf[0][0]
            #print('Diff image: ')
            show_image(x[0][0] , kept_cf[0][0] , diff)
            print('Predicted: ', model(torch.Tensor(kept_cf)).argmax(), ' || ', 'Desired: ', desired_class, ' || ', 'Orginal: ',  model(x).argmax())
            print('Sparsity = ', (diff).float().mean())
            print()