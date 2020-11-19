import os
import sys
import copy
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import timm
import matplotlib.pyplot as plt

from utils.drivers import test, get_dataloader, replace_conv_with_mconv
from pruner.fp_mbnetv3 import FilterPrunerMBNetV3
from pruner.fp_resnet import FilterPrunerResNet
from pruner.fp_resnet_cifar import FilterPrunerResNetCIFAR

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels import GridInterpolationKernel, AdditiveStructureKernel
from gpytorch.priors.torch_priors import GammaPrior
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.utils import standardize

from model.resnet_cifar10 import ResNet20, ResNet32, ResNet44, ResNet56, ResNet110

from model.resnet import resnet50, resnet18

from math import cos, pi
from torch.utils.tensorboard import SummaryWriter

import distributed as dist

import PIL

writer = None

def masked_forward(self, input, output):
    return (output.permute(0,2,3,1) * self.mask).permute(0,3,1,2)


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon = 0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(1).mean()
        return loss

class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = F.softmax(target,dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob).mean()
        return cross_entropy_loss

def set_lr(optim, lr):
    for params_group in optim.param_groups:
        params_group['lr'] = lr

def calculate_lr(initlr, cur_step, total_steps, warmup_steps):
    if cur_step < warmup_steps:
        curr_lr = initlr * (cur_step / warmup_steps)
    else:
        if args.scheduler == 'cosine_decay':
            N = (total_steps-warmup_steps)
            T = (cur_step - warmup_steps)
            curr_lr = initlr * (1 + cos(pi * T / (N-1))) / 2
        elif args.scheduler == 'linear_decay':
            N = (total_steps-warmup_steps)
            T = (cur_step - warmup_steps)
            curr_lr = initlr * (1-(float(T)/N))
    return curr_lr

class Hyperparams(object):
    def __init__(self, network):
        self.num_levels = 3
        self.cur_level = 1
        self.last_level = 0
        if args.network in ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110']:
            self.dim = [1, 6, int(args.network[6:])/2+2]

    def get_dim(self):
        return int(self.dim[self.cur_level-1])
    
    def random_sample(self):
        return np.random.rand(self.dim[self.cur_level-1]) * (args.upper_channel-args.lower_channel) + args.lower_channel
    
    def increase_level(self):
        if self.cur_level < self.num_levels:
            self.last_level = self.cur_level
            self.cur_level += 1
            return True
        return False
    
    def get_layer_budget_from_parameterization(self, parameterization, mask_pruner, soft=False):
        if not soft:
            parameterization = torch.tensor(parameterization)
            layers = len(mask_pruner.filter_ranks)

        layer_budget = torch.zeros(layers).cuda()
        if self.cur_level == 1:
            for k in range(layers):
                if mask_pruner.unit[k] == 1:
                    layer_budget[k] = torch.clamp(parameterization[0]*mask_pruner.filter_ranks[k].size(0), 1, mask_pruner.filter_ranks[k].size(0))
                else:
                    quant = (parameterization[0]*mask_pruner.filter_ranks[k].size(0) // mask_pruner.unit[k])*mask_pruner.unit[k]
                    layer_budget[k] = torch.clamp(quant, 1, mask_pruner.filter_ranks[k].size(0))

        elif self.cur_level == 2:
            if args.network in ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110']:
                depth = int(args.network[6:])
                stage = (depth - 2) // 3
                splits = [1]
                splits.extend([s*stage+1 for s in range(1,4)])
                for s in range(3):
                    for k in range(splits[s], splits[s+1], 2):
                        layer_budget[k] = torch.clamp(parameterization[s]*mask_pruner.filter_ranks[k].size(0), 1, mask_pruner.filter_ranks[k].size(0))
                splits = np.array(splits)+1
                splits[0] = 0
                last_left = 0
                last_filter = 0
                for s in range(3):
                    for k in range(splits[s], splits[s+1], 2):
                        layer_budget[k] = torch.clamp(parameterization[s+3]*mask_pruner.filter_ranks[k].size(0), 1, mask_pruner.filter_ranks[k].size(0))
                        layer_budget[k] = torch.clamp(last_left+parameterization[s+3]*(mask_pruner.filter_ranks[k].size(0)-last_filter), 1, mask_pruner.filter_ranks[k].size(0))
                        if k == (splits[s+1]-2):
                            last_left = layer_budget[k]
                            last_filter = mask_pruner.filter_ranks[k].size(0)
            else:
                lower = 0
                for p, upper in enumerate(mask_pruner.stages):
                    for k in range(lower, upper+1):
                        if mask_pruner.unit[k] == 1:
                            layer_budget[k] = torch.clamp(parameterization[p]*mask_pruner.filter_ranks[k].size(0), 1, mask_pruner.filter_ranks[k].size(0))
                        else:
                            quant = (parameterization[p]*mask_pruner.filter_ranks[k].size(0) // mask_pruner.unit[k])*mask_pruner.unit[k]
                            layer_budget[k] = torch.clamp(quant, 1, mask_pruner.filter_ranks[k].size(0))
                    lower = upper+1

        else:
            if args.network in ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110']:
                depth = int(args.network[6:])
                for k in range(1, depth-2, 2):
                    layer_budget[k] = torch.clamp(parameterization[(k-1)//2]*mask_pruner.filter_ranks[k].size(0), 1, mask_pruner.filter_ranks[k].size(0))

                stage = (depth - 2) // 3
                splits = [1]
                splits.extend([s*stage+1 for s in range(1,4)])
                splits = np.array(splits)+1
                splits[0] = 0
                last_left = 0
                last_filter = 0
                for s in range(3):
                    for k in range(splits[s], splits[s+1], 2):
                        layer_budget[k] = torch.clamp(parameterization[s-3]*mask_pruner.filter_ranks[k].size(0), 1, mask_pruner.filter_ranks[k].size(0))
                        layer_budget[k] = torch.clamp(last_left+parameterization[s+3]*(mask_pruner.filter_ranks[k].size(0)-last_filter), 1, mask_pruner.filter_ranks[k].size(0))
                        if k == (splits[s+1]-2):
                            last_left = layer_budget[k]
                            last_filter = mask_pruner.filter_ranks[k].size(0)

            else:
                p = 0
                for l in range(len(mask_pruner.filter_ranks)):
                    k = l
                    while k in mask_pruner.chains and layer_budget[k] == 0:
                        if mask_pruner.unit[k] == 1:
                            layer_budget[k] = torch.clamp(parameterization[p]*mask_pruner.filter_ranks[k].size(0), 1, mask_pruner.filter_ranks[k].size(0))
                        else:
                            quant = (parameterization[p]*mask_pruner.filter_ranks[k].size(0) // mask_pruner.unit[k])*mask_pruner.unit[k]
                            layer_budget[k] = torch.clamp(quant, 1, mask_pruner.filter_ranks[k].size(0))

                        k = mask_pruner.chains[k]
                    if layer_budget[k] == 0:
                        if mask_pruner.unit[k] == 1:
                            layer_budget[k] = torch.clamp(parameterization[p]*mask_pruner.filter_ranks[k].size(0), 1, mask_pruner.filter_ranks[k].size(0))
                        else:
                            quant = (parameterization[p]*mask_pruner.filter_ranks[k].size(0) // mask_pruner.unit[k])*mask_pruner.unit[k]
                            layer_budget[k] = torch.clamp(quant, 1, mask_pruner.filter_ranks[k].size(0))
                        p += 1

        if not soft:
            layer_budget = layer_budget.detach().cpu().numpy()
            for k in range(len(layer_budget)):
                layer_budget[k] = int(layer_budget[k])

        return layer_budget


class RandAcquisition(AcquisitionFunction):
    def setup(self, obj1, obj2, multiplier=None):
        self.obj1 = obj1
        self.obj2 = obj2
        self.rand = torch.rand(1) if multiplier is None else multiplier

    def forward(self, X):
        linear_weighted_sum = (1-self.rand) * (self.obj1(X)-args.baseline) + self.rand * (self.obj2(X)-args.baseline)
        # NOTE: This is just the augmented Tchebyshev scalarization in the paper
        return -1*(torch.max((1-self.rand) * (self.obj1(X)-args.baseline), self.rand * (self.obj2(X)-args.baseline)) + (1e-6 * linear_weighted_sum))


def is_pareto_efficient(costs, return_mask = True, epsilon=0):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    # NOTE: This is the non-dominating sort
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index]-epsilon, axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

class MOFP:
    def __init__(self, dataset, datapath, model, pruner, sample_pool=None, rank_type='l2_weight', batch_size=32, safeguard=0, global_random_rank=False, lub='', device='cuda', resource='filter'):
        self.device = device
        self.sample_for_ranking = 1 if rank_type in ['l1_weight', 'l2_weight', 'l0_weight', 'l2_bn', 'l1_bn', 'l2_bn_param'] else 5000
        self.safeguard = safeguard
        self.lub = lub
        self.img_size = 32 if 'CIFAR' in args.dataset else 224
        self.batch_size = batch_size
        self.rank_type = rank_type
    
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(self.img_size, dataset, datapath, batch_size, eval(args.interpolation), True, args.slim_dataaug, args.scale_ratio, num_gpus=dist.get_world_size())

        if 'CIFAR100' in dataset:
            num_classes = 100
        elif 'CIFAR10' in dataset:
            num_classes = 10
        elif 'ImageFolder' in dataset:
            num_classes = 1000
        self.num_classes = num_classes
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mask_pruner = eval(pruner)(self.model, rank_type, num_classes, safeguard, random=global_random_rank, device=device, resource='flops') 
        self.use_mem = resource == 'mem'

        self.model.train()

        self.sample_pool = None
        if sample_pool is not None:
            self.sample_pool = torch.load(sample_pool)['X']

        self.sampling_weights = np.ones(50)

    def sample_arch(self, START_BO, g, steps, hyperparams, og_flops, full_val_loss, target_flops=0):
        if args.slim:
            if self.sample_pool is not None:
                idx = np.random.choice(len(self.sample_pool), 1)[0]
                parameterization = self.sample_pool[idx]
            else:
                if target_flops == 0:
                    parameterization = hyperparams.random_sample()
                else:
                    parameterization = np.ones(hyperparams.get_dim()) * args.lower_channel

            layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
        else:
            if g < START_BO:
                if self.sample_pool is not None:
                    idx = np.random.choice(len(self.sample_pool), 1)[0]
                    parameterization = self.sample_pool[idx]
                else:
                    if target_flops == 0:
                        f = np.random.rand(1) * (args.upper_channel-args.lower_channel) + args.lower_channel
                    else:
                        f = args.lower_channel
                    parameterization = np.ones(hyperparams.get_dim()) * f
                layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
            elif g == START_BO:
                if target_flops == 0:
                    parameterization = np.ones(hyperparams.get_dim())
                else:
                    f = args.lower_channel
                    parameterization = np.ones(hyperparams.get_dim()) * f
                layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
            else:
                rand = torch.rand(1).cuda(self.device)

                train_X = torch.FloatTensor(self.X).cuda(self.device)
                train_Y_loss = torch.FloatTensor(np.array(self.Y)[:, 0].reshape(-1, 1)).cuda(self.device)
                train_Y_loss = standardize(train_Y_loss)

                train_Y_cost = torch.FloatTensor(np.array(self.Y)[:, 1].reshape(-1, 1)).cuda(self.device)
                train_Y_cost = standardize(train_Y_cost)

                covar_module = None
                if args.ski and g > 128:
                    if args.additive:
                        covar_module = AdditiveStructureKernel(
                            ScaleKernel(
                                GridInterpolationKernel(
                                    MaternKernel(
                                        nu=2.5,
                                        lengthscale_prior=GammaPrior(3.0, 6.0),
                                    ),
                                    grid_size=128, num_dims=1, grid_bounds=[(0, 1)]
                                ),
                                outputscale_prior=GammaPrior(2.0, 0.15),
                            ), 
                            num_dims=train_X.shape[1]
                        )
                    else:
                        covar_module = ScaleKernel(
                            GridInterpolationKernel(
                                MaternKernel(
                                    nu=2.5,
                                    lengthscale_prior=GammaPrior(3.0, 6.0),
                                ),
                                grid_size=128, num_dims=train_X.shape[1], grid_bounds=[(0, 1) for _ in range(train_X.shape[1])]
                            ),
                            outputscale_prior=GammaPrior(2.0, 0.15),
                        )
                else:
                    if args.additive:
                        covar_module = AdditiveStructureKernel(
                            ScaleKernel(
                                MaternKernel(
                                    nu=2.5,
                                    lengthscale_prior=GammaPrior(3.0, 6.0),
                                    num_dims=1
                                ),
                                outputscale_prior=GammaPrior(2.0, 0.15),
                            ),
                            num_dims=train_X.shape[1]
                        )
                    else:
                        covar_module = ScaleKernel(
                            MaternKernel(
                                nu=2.5,
                                lengthscale_prior=GammaPrior(3.0, 6.0),
                                num_dims=train_X.shape[1]
                            ),
                            outputscale_prior=GammaPrior(2.0, 0.15),
                        )

                new_train_X = train_X
                gp_loss = SingleTaskGP(new_train_X, train_Y_loss, covar_module=covar_module)
                mll = ExactMarginalLogLikelihood(gp_loss.likelihood, gp_loss)
                mll = mll.to(self.device)
                fit_gpytorch_model(mll)


                # Use add-gp for cost
                covar_module = AdditiveStructureKernel(
                    ScaleKernel(
                        MaternKernel(
                            nu=2.5,
                            lengthscale_prior=GammaPrior(3.0, 6.0),
                            num_dims=1
                        ),
                        outputscale_prior=GammaPrior(2.0, 0.15),
                    ),
                    num_dims=train_X.shape[1]
                )
                gp_cost = SingleTaskGP(new_train_X, train_Y_cost, covar_module=covar_module)
                mll = ExactMarginalLogLikelihood(gp_cost.likelihood, gp_cost)
                mll = mll.to(self.device)
                fit_gpytorch_model(mll)

                UCB_loss = UpperConfidenceBound(gp_loss, beta=args.beta).cuda(self.device)
                UCB_cost = UpperConfidenceBound(gp_cost, beta=args.beta).cuda(self.device)
                self.mobo_obj = RandAcquisition(UCB_loss).cuda(self.device)
                self.mobo_obj.setup(UCB_loss, UCB_cost, rand)

                lower = torch.ones(new_train_X.shape[1])*args.lower_channel
                upper = torch.ones(new_train_X.shape[1])*args.upper_channel
                self.mobo_bounds = torch.stack([lower, upper]).cuda(self.device)

                if args.pas:
                    # NOTE: uniformly sample FLOPs
                    val = np.linspace(args.lower_flops, 1, 50)
                    chosen_target_flops = np.random.choice(val, p=(self.sampling_weights/np.sum(self.sampling_weights)))
                    
                    lower_bnd, upper_bnd = 0, 1
                    lmda = 0.5
                    for i in range(10):
                        self.mobo_obj.rand = lmda

                        parameterization, acq_value = optimize_acqf(
                            self.mobo_obj, bounds=self.mobo_bounds, q=1, num_restarts=5, raw_samples=1000,
                        )

                        parameterization = parameterization[0].cpu().numpy()

                        # NOTE: quantize to multiple of 0.1
                        parameterization = np.round(parameterization*10)/10
                        parameterization = np.clip(parameterization, args.lower_channel, args.upper_channel)

                        layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
                        sim_flops = self.mask_pruner.simulate_and_count_flops(layer_budget, self.use_mem)
                        ratio = sim_flops/og_flops

                        if np.abs(ratio - chosen_target_flops) <= 0.02:
                            break
                        if args.baseline > 0:
                            if ratio < chosen_target_flops:
                                lower_bnd = lmda
                                lmda = (lmda + upper_bnd) / 2
                            elif ratio > chosen_target_flops:
                                upper_bnd = lmda
                                lmda = (lmda + lower_bnd) / 2
                        else:
                            if ratio < chosen_target_flops:
                                upper_bnd = lmda
                                lmda = (lmda + lower_bnd) / 2
                            elif ratio > chosen_target_flops:
                                lower_bnd = lmda
                                lmda = (lmda + upper_bnd) / 2
                    rand[0] = lmda
                    if dist.is_master():
                        writer.add_scalar('Binary search trials', i, steps)

                else:
                    parameterization, acq_value = optimize_acqf(
                        self.mobo_obj, bounds=self.mobo_bounds, q=1, num_restarts=5, raw_samples=1000,
                    )
                    parameterization = parameterization[0].cpu().numpy()

                    # NOTE: quantize to multiple of 0.1
                    parameterization = np.round(parameterization*10)/10

                layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
        return layer_budget, parameterization, self.sampling_weights/np.sum(self.sampling_weights)

    def channels_to_mask(self, layer_budget):
        prune_targets = []
        for k in sorted(self.mask_pruner.filter_ranks.keys()):
            if (self.mask_pruner.filter_ranks[k].size(0) - layer_budget[k]) > 0:
                prune_targets.append((k, (int(layer_budget[k]), self.mask_pruner.filter_ranks[k].size(0) - 1)))
        return prune_targets

    def bo_share(self):
        START_BO = args.prior_points
        self.population_data = []

        self.mask_pruner.reset() 
        self.mask_pruner.model.eval()
        self.mask_pruner.forward(torch.zeros((1,3,self.img_size,self.img_size), device=self.device))

        if args.distill:
            self.teacher = copy.deepcopy(self.mask_pruner.model)


        # Optimizer
        iters_per_epoch = len(self.train_loader)
        ### all parameter ####
        no_wd_params, wd_params = [], []
        for name, param in self.mask_pruner.model.named_parameters():
            if param.requires_grad:
                if ".bn" in name or '.bias' in name:
                    no_wd_params.append(param)
                else:
                    wd_params.append(param)
        no_wd_params = nn.ParameterList(no_wd_params)
        wd_params = nn.ParameterList(wd_params)
        lr = args.baselr * (args.batch_size / 256.)

        if args.warmup > 0:
            optimizer = torch.optim.SGD([
                            {'params': no_wd_params, 'weight_decay':0.},
                            {'params': wd_params, 'weight_decay': args.wd},
                        ], lr/float(iters_per_epoch*args.warmup), momentum=args.mmt, nesterov=args.nesterov)
        else:
            optimizer = torch.optim.SGD([
                            {'params': no_wd_params, 'weight_decay':0.},
                            {'params': wd_params, 'weight_decay': args.wd},
                        ], lr, momentum=args.mmt, nesterov=args.nesterov)
        lrinfo = {'initlr': lr, 'warmup_steps': args.warmup*iters_per_epoch,
                'total_steps': args.epochs*iters_per_epoch}

        criterion = CrossEntropyLabelSmooth(self.num_classes, args.label_smoothing).to(self.device)
        kd = CrossEntropyLossSoft().cuda(self.device)

        self.mask_pruner.reset() 
        self.mask_pruner.model.eval()
        self.mask_pruner.forward(torch.zeros((1,3,self.img_size,self.img_size), device=self.device))

        ind_layers = 0
        checked = np.zeros(len(self.mask_pruner.filter_ranks))
        for l in sorted(self.mask_pruner.filter_ranks.keys()):
            if checked[l]:
                continue
            k = l
            while k in self.mask_pruner.chains:
                k = self.mask_pruner.chains[k]
                checked[k] = 1
            ind_layers += 1
                
        hyperparams = Hyperparams(args.network)
        if args.network not in ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110']:
            hyperparams.dim = [1, len(self.mask_pruner.stages), ind_layers]
        for _ in range(args.param_level-1):
            hyperparams.increase_level()

        parameterization = np.ones(hyperparams.get_dim())
        layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
        og_flops = self.mask_pruner.simulate_and_count_flops(layer_budget, self.use_mem)

        if args.lower_channel != 0:
            parameterization = np.ones(hyperparams.get_dim()) * args.lower_channel
            layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
            sim_flops = self.mask_pruner.simulate_and_count_flops(layer_budget, self.use_mem)
            args.lower_flops = (float(sim_flops) / og_flops)
            print('Lower flops based on lower channel: {}'.format(args.lower_flops))
        print('Full MFLOPs: {:.3f}'.format(og_flops/1e6))


        self.X = None
        self.Y = []

        self.mask_pruner.model.train()

        if args.distill:
            with torch.no_grad():
                test_top1, test_top5 = test(self.teacher, self.test_loader, device=self.device)
                print('Pre-trained | Top-1: {:.2f}, Top-5: {:.2f}'.format(test_top1, test_top5))

        
        og_filters = []
        for k in sorted(self.mask_pruner.filter_ranks.keys()):
            og_filters.append(self.mask_pruner.filter_ranks[k].size(0))

        g = 0
        start_epoch = 0
        maxloss = 0
        minloss = 0
        ratio_visited = []
        archs = []

        if os.path.exists(os.path.join('./checkpoint/', '{}.pt'.format(args.name))):
            ckpt = torch.load(os.path.join('./checkpoint/', '{}.pt'.format(args.name)))
            self.X = ckpt['X']
            self.Y = ckpt['Y']
            self.population_data = ckpt['population_data']
            self.mask_pruner.model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optim_state_dict'])
            start_epoch = ckpt['epoch']+1
            if len(self.population_data) > 1:
                g = len(self.X)
                archs = [data['filters'] for data in self.population_data[-args.num_sampled_arch:]]
            if 'ratio_visited' in ckpt:
                ratio_visited = ckpt['ratio_visited']
            if dist.is_master():
                print('Loading checkpoint from epoch {}'.format(start_epoch-1))


        full_val_loss = 0
        if args.freeze_mobo:
            for j in range(len(self.Y)):
                if self.Y[j][1] == 1:
                    full_val_loss = self.Y[j][0]
                if self.Y[j][0] == 0:
                    break
            self.Y = self.Y[:j]
            self.X = self.X[:j]
            self.population_data = self.population_data[:j]

        # Sync-BN
        if args.sync_bn:
            self.mask_pruner.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.mask_pruner.model)
            self.mask_pruner.model.cuda(self.device)
        self.mask_pruner.model = torch.nn.parallel.DistributedDataParallel(
            module=self.mask_pruner.model, device_ids=[self.device], output_device=self.device
        )

        for epoch in range(start_epoch, args.epochs):
            self.train_loader.sampler.set_epoch(epoch)
            start_time = time.time()
            for i, (batch, label) in enumerate(self.train_loader):
                self.mask_pruner.model.train()
                cur_step = iters_per_epoch*epoch+i
                lr = calculate_lr(lrinfo['initlr'], cur_step, lrinfo['total_steps'], lrinfo['warmup_steps'])
                set_lr(optimizer, lr)
                batch, label = batch.to(self.device), label.to(self.device)

                if not args.normal_training:
                    if not args.slim:
                        if cur_step % args.tau == 0:
                            # NOTE: Calibration of historical data
                            if len(self.Y) > 1 and not args.freeze_mobo:
                                diff = 0
                                for j in range(len(self.Y)):
                                    with torch.no_grad():
                                        self.remove_mask()
                                        self.mask(self.population_data[j]['filters'])
                                        output = self.mask_pruner.model(batch)
                                        loss = criterion(output, label).item()

                                        if self.Y[j][1] == 1:
                                            full_val_loss = loss

                                        diff += np.abs(loss - self.Y[j][0])
                                        self.Y[j][0] = loss
                                        self.population_data[j]['loss'] = loss

                    if cur_step % args.tau == 0:
                        archs = []
                        ratios = []
                        sampled_sim_flops = []
                        parameterizations = []
                        # Sample architecture
                        for _ in range(args.num_sampled_arch):
                            layer_budget, parameterization, weights = self.sample_arch(START_BO, g, cur_step, hyperparams, og_flops, full_val_loss)
                            if not args.slim:
                                sim_flops = self.mask_pruner.simulate_and_count_flops(layer_budget, self.use_mem)
                                sampled_sim_flops.append(sim_flops)
                                ratio = sim_flops/og_flops
                                ratios.append(ratio)
                                ratio_visited.append(ratio)

                                parameterizations.append(parameterization)
                                g += 1

                            prune_targets = self.channels_to_mask(layer_budget)
                            archs.append(prune_targets)

                        if not args.slim:
                            if self.X is None:
                                self.X = np.array(parameterizations)
                            else:
                                self.X = np.concatenate([self.X, parameterizations], axis=0)
                            for ratio, sim_flops, prune_targets in zip(ratios, sampled_sim_flops, archs):
                                self.Y.append([0, ratio])
                                self.population_data.append({'loss': 0, 'flops': sim_flops, 'ratio': ratio, 'filters': prune_targets})


                        # Smallest model
                        parameterization = np.ones(hyperparams.get_dim()) * args.lower_channel
                        layer_budget = hyperparams.get_layer_budget_from_parameterization(parameterization, self.mask_pruner)
                        sim_flops = self.mask_pruner.simulate_and_count_flops(layer_budget, self.use_mem)
                        ratio = sim_flops/og_flops
                        prune_targets = self.channels_to_mask(layer_budget)
                        archs.append(prune_targets)

                # Inplace distillation
                self.mask_pruner.model.zero_grad()
                if args.distill:
                    with torch.no_grad():
                        t_output = self.teacher(batch)
                    self.remove_mask()
                    output = self.mask_pruner.model(batch)
                    loss = kd(output, t_output.detach())
                    loss.backward()
                    loss = dist.scaled_all_reduce([loss])[0]
                    maxloss = loss.item()
                else:
                    self.remove_mask()
                    t_output = self.mask_pruner.model(batch)
                    loss = criterion(t_output, label)
                    loss.backward()
                    loss = dist.scaled_all_reduce([loss])[0]
                    maxloss = loss.item()
                for prune_targets in archs:
                    self.remove_mask()
                    self.mask(prune_targets)
                    output = self.mask_pruner.model(batch)
                    loss = kd(output, t_output.detach())
                    loss.backward()
                    loss = dist.scaled_all_reduce([loss])[0]
                    minloss = loss.item()

                if dist.is_master() and cur_step % args.print_freq == 0:
                    for param_group in optimizer.param_groups:
                        lr = param_group['lr']
                    writer.add_scalar('Loss for largest model', maxloss, epoch*len(self.train_loader)+i)
                    writer.add_scalar('Loss for smallest model', minloss, epoch*len(self.train_loader)+i)
                    writer.add_scalar('Learning rate', lr, epoch*len(self.train_loader)+i)

                dist.allreduce_grads(model)
                optimizer.step()
                sys.stdout.flush()

            if dist.is_master():
                if not os.path.exists('./checkpoint/'):
                    os.makedirs('./checkpoint/')
                torch.save({'model_state_dict': self.mask_pruner.model.state_dict(), 'optim_state_dict': optimizer.state_dict(),
                            'epoch': epoch, 'population_data': self.population_data, 'X': self.X, 'Y': self.Y, 'ratio_visited': ratio_visited}, os.path.join('./checkpoint/', '{}.pt'.format(args.name)))
                if len(ratio_visited) > 0:
                    writer.add_histogram('FLOPs visited', np.array(ratio_visited), epoch+1)
                print('Epoch {} | Time: {:.2f}s'.format(epoch, time.time()-start_time))

            if args.normal_training:
                test_top1, test_top5 = test(self.mask_pruner.model, self.test_loader, device=self.device)
                if dist.is_master():
                    writer.add_scalar('Test acc/Top-1', test_top1, epoch+1)
                    writer.add_scalar('Test acc/Top-5', test_top1, epoch+1)

    def mask(self, prune_targets):
        for layer_index, filter_index in prune_targets:
            self.mask_pruner.activation_to_conv[layer_index].mask[filter_index[0]:filter_index[1]+1].zero_()

    def remove_mask(self):
        for k in sorted(self.mask_pruner.filter_ranks.keys()):
            self.mask_pruner.activation_to_conv[k].mask.zero_()
            self.mask_pruner.activation_to_conv[k].mask += 1

def get_args():
    parser = argparse.ArgumentParser()
    # Configuration
    parser.add_argument("--name", type=str, default='test', help='Name for the experiments, the resulting model and logs will use this')
    parser.add_argument("--datapath", type=str, default='./data', help='Path toward the dataset that is used for this experiment')
    parser.add_argument("--dataset", type=str, default='torchvision.datasets.CIFAR10', help='The class name of the dataset that is used, please find available classes under the dataset folder')
    parser.add_argument("--model", type=str, default='', help='The pre-trained model that pruning starts from')
    parser.add_argument("--mask_model", type=str, default='', help='The model used to derive mask')
    parser.add_argument("--network", type=str, default='Conv6', help='The model used to derive mask')
    parser.add_argument("--reinit", action='store_true', default=False, help='Not using pre-trained models, has to be specified for re-training timm models')
    parser.add_argument("--resource_type", type=str, default='filter', help='determining the threshold')
    parser.add_argument("--pruner", type=str, default='FilterPrunerMBNetV3', help='Different network require differnt pruner implementation')
    parser.add_argument("--rank_type", type=str, default='l2_weight', help='The ranking criteria for filter pruning')
    parser.add_argument("--global_random_rank", action='store_true', default=False, help='When this is specified, none of the rank_type matters, it will randomly prune the filters')
    parser.add_argument("--width_mult", type=float, default=1., help='Width multiplier')
    parser.add_argument("--interpolation", type=str, default='PIL.Image.BILINEAR', help='Image resizing interpolation')
    parser.add_argument("--print_freq", type=int, default=500, help='Logging frequency in iterations')

    # Training
    parser.add_argument("--epochs", type=int, default=120, help='Number of training epochs')
    parser.add_argument("--warmup", type=int, default=5, help='Number of warmup epochs')
    parser.add_argument("--baselr", type=float, default=0.05, help='The learning rate for fine-tuning')
    parser.add_argument("--scheduler", type=str, default='cosine_decay', help='Support: cosine_decay | linear_decay')
    parser.add_argument("--mmt", type=float, default=0.9, help='Momentum for fine-tuning')
    parser.add_argument("--tau", type=int, default=200, help='training iterations for one architecture')
    parser.add_argument("--wd", type=float, default=1e-4, help='The weight decay used')
    parser.add_argument("--scale_ratio", type=float, default=0.08, help='Scale for random scaling, default: 0.08')
    parser.add_argument("--label_smoothing", type=float, default=1e-1, help='Label smoothing')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size for training')
    parser.add_argument("--distill", action='store_true', default=False, help='Distillation from pre-trained model')
    parser.add_argument("--logging", action='store_true', default=False, help='Log the output')
    parser.add_argument("--normal_training", action='store_true', default=False, help='For independent trained model')
    parser.add_argument("--nesterov", action='store_true', default=False, help='For independent trained model')
    parser.add_argument("--slim_dataaug", action='store_true', default=False, help='Use the data augmentation implemented in universally slimmable network')
    parser.add_argument("--seed", type=int, default=0, help='Random seed')

    # Channel
    parser.add_argument("--safeguard", type=float, default=0, help='A floating point number that represent at least how many percentage of the original number of channel should be preserved. E.g., 0.10 means no matter what ranking, each layer should have at least 10% of the number of original channels.')
    parser.add_argument("--param_level", type=int, default=1, help='Level of parameterization')
    parser.add_argument("--lower_channel", type=float, default=0, help='lower bound')
    parser.add_argument("--upper_channel", type=float, default=1, help='upper bound')
    parser.add_argument("--lower_flops", type=float, default=0.1, help='lower bound')
    parser.add_argument("--upper_flops", type=float, default=1, help='upper bound')
    parser.add_argument("--slim", action='store_true', default=False, help='Use slimmable training')
    parser.add_argument("--no_channel_proj", action='store_true', default=False, help='Use Adam instead of SGD')
    parser.add_argument("--num_sampled_arch", type=int, default=1, help='Number of arch sampled in between largest and smallest')
    parser.add_argument('--track_flops', nargs='+', default=[0.35, 0.5, 0.75])
    parser.add_argument("--cont_sampling", action='store_true', default=False, help='Continuous sampling previous arch to train')
    parser.add_argument("--sample_pool", type=str, default='none', help='Checkpoint to sample architectures from')
    parser.add_argument("--sync_bn", action='store_true', default=False, help='Use sync bn')

    # GP-related hyper-param (PareCO)
    parser.add_argument("--buffer", type=int, default=1000, help='Buffer for GP')
    parser.add_argument("--beta", type=float, default=0.1, help='For UCB')
    parser.add_argument("--prior_points", type=int, default=10, help='Number of uniform arch for BO')
    parser.add_argument("--baseline", type=int, default=5, help='Use for scalarization')
    parser.add_argument("--ski", action='store_true', default=False, help='Use Adam instead of SGD')
    parser.add_argument("--additive", action='store_true', default=False, help='Use Adam instead of SGD')
    parser.add_argument("--pas", action='store_true', default=False, help='Pareto-aware scalarization')
    parser.add_argument("--freeze_mobo", action='store_true', default=False, help='Freeze the historical data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # dist.init_dist(False)
    if not os.path.exists('./config/'):
        os.makedirs('./config/')
    dist.init_dist()
    args = get_args()
    random_seed = 3080 + args.seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    if dist.is_master():
        writer = SummaryWriter('./runs/{}'.format(args.name))
        if args.logging:
            if not os.path.exists('./log'):
                os.makedirs('./log')
            sys.stdout = open('./log/{}.log'.format(args.name), 'a')
    if dist.is_master():
        print(args)

    if 'CIFAR100' in args.dataset:
        num_classes = 100
    elif 'CIFAR10' in args.dataset:
        num_classes = 10
    elif 'ImageFolder' in args.dataset:
        num_classes = 1000

    device = torch.cuda.current_device()

    if args.network == 'mobilenetv3':
        model = timm.create_model('mobilenetv3_large_100', pretrained=not args.reinit)
    elif args.network in ['mobilenetv2_035', 'mobilenetv2_050', 'mobilenetv2_075', 'mobilenetv2_100']:
        model = timm.create_model(args.network, pretrained=not args.reinit)
    else:
        model = eval(args.network)(num_classes=num_classes)

    replace_conv_with_mconv(model)
    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d):
    #         m.register_forward_hook(masked_forward)
    #         m.mask = nn.Parameter(torch.ones(m.weight.size(0)), requires_grad=False)
    model = model.to(device)

    sample_pool = None if args.sample_pool == 'none' else args.sample_pool
    mofp = MOFP(args.dataset, args.datapath, model, args.pruner, sample_pool, args.rank_type, args.batch_size, safeguard=args.safeguard, global_random_rank=args.global_random_rank, device=device, resource=args.resource_type)

    mofp.bo_share()
