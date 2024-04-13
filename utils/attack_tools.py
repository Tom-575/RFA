import torch
import torch.nn as nn
import random
import numpy as np
from math import inf
from advertorch.attacks import *
import sys
# from packages.PerceptualAdvex.perceptual_advex.attacks import StAdvAttack, ReColorAdvAttack


# advertorch.attacks
def FGSMAttack(model, eps=8./255, targeted=False):
    return GradientSignAttack(model, eps=eps, clip_min=0.0, clip_max=1.0, targeted=targeted)

def CWL2Attack(model,num_classes, num_iters=40, targeted=False, initial_const=1):
    return CarliniWagnerL2Attack(model, num_classes, clip_min=0.0, clip_max=1.0, max_iterations=num_iters,
        confidence=1, initial_const=initial_const, learning_rate=1e-2, binary_search_steps=4, targeted=targeted)

def PGDLinfAttack(model, num_iters=40, eps=8./255, targeted=False):
    return LinfPGDAttack(model, eps=eps, nb_iter=num_iters, eps_iter=eps/10, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=targeted)

def PGDL2Attack(model, num_iters=40, eps=1.0, targeted=False):
    return L2PGDAttack(model, eps=eps, nb_iter=num_iters, eps_iter=eps/10, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=targeted)

def BIMLinfAttack(model, num_iters=40, eps=8./255, targeted=False):
    return LinfBasicIterativeAttack(model, eps=eps, nb_iter=num_iters, eps_iter=eps/10, clip_min=0., clip_max=1., targeted=targeted)

def BIML2Attack(model, num_iters=40, eps=1.0, targeted=False):
    return L2BasicIterativeAttack(model, eps=eps, nb_iter=num_iters, eps_iter=eps/10, clip_min=0., clip_max=1., targeted=targeted)

def DFLinfAttack(model, num_classes, num_iters=40, eps=8./255, targeted=False):
    return DeepfoolLinfAttack(model, num_classes, nb_iter=num_iters, clip_min=0., clip_max=1., targeted=targeted)

def SpaTransformAttack(model, num_classes, max_iterations=100):
    return SpatialTransformAttack(model, num_classes, confidence=0, initial_const=1, max_iterations=max_iterations, \
                search_steps=1, loss_fn=None, clip_min=0.0, clip_max=1.0, abort_early=True, targeted=False)
 

# perceptual_advex.attacks
DATASET_NUM_CLASSES = {
    'cifar': 10,
    'cifar10': 10,
    'cifar100': 100,
    'imagenet100': 100,
    'imagenet': 1000,
    # 'bird_or_bicycle': 2,
}

class AutoAttack(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()

        kwargs.setdefault('verbose', False)
        self.model = model
        self.kwargs = kwargs
        self.attack = None

    def forward(self, inputs, labels):
        # Necessary to initialize attack here because for parallelization
        # across multiple GPUs.
        if self.attack is None:
            try:
                import autoattack
            except ImportError:
                raise RuntimeError(
                    'Error: unable to import autoattack. Please install the '
                    'package by running '
                    '"pip install git+git://github.com/fra31/auto-attack#egg=autoattack".'
                )
            self.attack = autoattack.AutoAttack(
                self.model, device=inputs.device, **self.kwargs)
        return self.attack.run_standard_evaluation(inputs, labels)

class AutoLinfAttack(AutoAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 8/255,
                'imagenet100': 8/255,
                'imagenet': 8/255,
                'bird_or_bicycle': 16/255,
            }[dataset_name]

        super().__init__(
            model,
            norm='Linf',
            eps=bound,
            **kwargs,
        )

class AutoL2Attack(AutoAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 1,
                'imagenet100': 3,
                'imagenet': 3,
                'bird_or_bicycle': 10,
            }[dataset_name]

        super().__init__(
            model,
            norm='L2',
            eps=bound,
            **kwargs,
        )


class NoAttack(nn.Module):
    """
    Attack that does nothing.
    """

    def __init__(self, model=None):
        super().__init__()
        self.model = model

    def forward(self, inputs, labels=None):
        return inputs

def Clean():
    return NoAttack()

def AALinfAttack(model, dataset_name='cifar10', bound=8/255,  version='standard'):
    return AutoLinfAttack(model, dataset_name=dataset_name, bound=bound, version=version)
    
def AAL2Attack(model, dataset_name='cifar10', bound=1.0, version='standard'):
    return AutoL2Attack(model, dataset_name=dataset_name, bound=bound,version=version)



# def STAAttack(model, num_iters=100, bound=0.05):
#     return StAdvAttack(model, num_iterations=num_iters, bound=bound)

# def ReColorAttack(model, num_iters=100, bound=0.06):
#     return ReColorAdvAttack(model, num_iterations=num_iters, bound=bound)






