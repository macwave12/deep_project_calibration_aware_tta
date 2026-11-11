import torch
from copy import deepcopy
import numpy as np
import operator
import math
from torch.nn import functional as F

def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).to(image_features.device).float() #.half()
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).to(image_features.device).float() # .half()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits
    
def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]

PROTOCOLS = ('published', 'clean')
# TDA's ImageNet-val setting (supp. Table 6), used on every dataset under 'clean'.
CLEAN_PROTOCOL = {'alpha': 2.0, 'beta': 5.0}


def resolve_protocol(args):
    """`args.protocol`, defaulting to 'published' when absent (old callers)."""
    protocol = getattr(args, 'protocol', None) or 'published'
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    return protocol


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

class TDA():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        self.pos_params = {
            'alpha': {'Caltech101':5.0,'DTD':2.0,'eurosat':4.0,'Food101':1.0,'Flower102':1.0,'Cars':1.0, 'UCF101':3.0, 'Pets':2.0, 'SUN397':2.0, 'Aircraft': 2.0, 'A':2.0, 'R':1.0, 'K':2.363, 'V':1.0, 'I':2.0, 'office-home':2.0, 'DomainNet':2.0},
            'beta':{'Caltech101':5.0,'DTD':3.0,'eurosat':8.0,'Food101':1.0,'Flower102':5.0,'Cars':7.0, 'UCF101':8.0, 'Pets':7.0, 'SUN397':3.0, 'Aircraft': 2.0, 'A':5.0, 'R':8.0, 'K':7.45, 'V':8.0, 'I':5.0, 'office-home':5.0, 'DomainNet':5.0}
        }

    def prepare_model_and_optimization(self, args):
        self.model.eval()

        self.use_pos_cache = True
        self.use_neg_cache = True

        # Hyperparameter protocol (pm/PLAN.md). 'published' (the default) keeps
        # the per-dataset table above; 'clean' uses TDA's single searched
        # setting on every dataset, so no test set informed the choice. Only the
        # positive cache's alpha/beta differ: the negative cache's are global.
        # Under 'clean' the table is not read, so datasets TDA never tuned
        # (the T15 fresh sets) run without an entry in it.
        self.protocol = resolve_protocol(args)
        if self.protocol == 'clean':
            self.pos_alpha = CLEAN_PROTOCOL['alpha']
            self.pos_beta = CLEAN_PROTOCOL['beta']
        else:
            self.pos_alpha = self.pos_params['alpha'][args.test_sets]
            self.pos_beta = self.pos_params['beta'][args.test_sets]

        self.neg_alpha = 0.117
        self.neg_beta = 1.0

        self.pos_cache = {}
        self.neg_cache = {}

    def pre_adaptation(self):
        pass
    
    def adaptation_process(self, image, images, args):

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image_features, text_features, logit_scale = self.model.forward_features(images)
                logits = logit_scale * image_features @ text_features.t()
                softmax0 = logits.softmax(dim=-1)

        ent0 = Entropy(softmax0)
        pred0 = torch.max(logits, 1)[1].item()
        num_classes = logits.size(1)

        update_cache(self.pos_cache, pred0, [image_features, ent0], 3)

        if 0.2 < ent0/math.log2(num_classes) and ent0/math.log2(num_classes) < 0.5:
            update_cache(self.neg_cache, pred0, [image_features, ent0, softmax0], 2, True)

        output = logits.clone()

        if self.use_pos_cache and len(self.pos_cache) > 0:
            output += compute_cache_logits(image_features, self.pos_cache, self.pos_alpha, self.pos_beta, text_features.unsqueeze(0))
        if self.use_neg_cache and len(self.neg_cache) > 0:
            output -= compute_cache_logits(image_features, self.neg_cache, self.neg_alpha, self.neg_beta, text_features.unsqueeze(0), (0.03, 1.0))

        return_dict = {
            'output':output,
        }

        return return_dict



