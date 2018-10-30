#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.sparse
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os, sys

import utils


class ESNCell(nn.Module):
    def __init__(
        self, input_dim, output_dim, batch_size, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0, w=None, w_in=None,
        w_bias=None, w_fdb=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None, leaky_rate=0.2,
        nonlin_func=torch.tanh, feedbacks=False, feedbacks_dim=None, wfdb_sparsity=None,
        normalize_feedbacks=False):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param output_dim: Reservoir size
        :param spectral_radius: Reservoir's spectral radius
        :param bias_scaling: Scaling of the bias, a constant input to each neuron (default: 0, no bias)
        :param input_scaling: Scaling of the input weight matrix, default 1.
        :param w: Internation weights matrix
        :param w_in: Input-reservoir weights matrix
        :param w_bias: Bias weights matrix
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func: Reservoir's activation function (tanh, sig, relu)
        """
        super(ESNCell, self).__init__()

        # Params
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.bias_scaling = bias_scaling
        self.input_scaling = input_scaling
        self.sparsity = sparsity
        self.input_set = input_set
        self.w_sparsity = w_sparsity
        self.leaky_rate = leaky_rate
        self.nonlin_func = nonlin_func
        self.feedbacks = feedbacks
        self.feedbacks_dim = feedbacks_dim
        self.wfdb_sparsity = wfdb_sparsity
        self.normalize_feedbacks = normalize_feedbacks
        # Init hidden state
        self.register_buffer('hidden', self.init_hidden(batch_size))
        # Initialize input weights
        self.register_buffer('w_in', self._generate_win(w_in))
        # Initialize reservoir weights randomly
        self.register_buffer('w', self._generate_w(w))
        # Initialize bias
        self.register_buffer('w_bias', self._generate_wbias(w_bias))
        # Initialize feedbacks weights randomly
        if feedbacks:
            self.register_buffer('w_fdb', self._generate_wfdb(w_fdb))
    

    def init_hidden(self, batch_size):
        return torch.zeros(self.output_dim).requires_grad_(requires_grad=False)

    def reset_hidden(self):
        self.hidden.fill_(0.0)

    # Get W's spectral radius
    def get_spectral_radius(self):
        return utils.spectral_radius(self.w)
    
    def _generate_w(self, w):
        """
        Generate W matrix
        :return:
        """
        if w is None:
            w = self.generate_w(self.output_dim, self.w_sparsity)
        else:
            if callable(w):
                w = w(self.output_dim)
        # Scale it to spectral radius
        w *= self.spectral_radius / utils.spectral_radius(w)
        return w.requires_grad_(requires_grad=False)
    

    def _generate_win(self, w_in):
        """
        Generate Win matrix
        :return:
        """
        if w_in is None:
            if self.sparsity is None:
                w_in = self.input_scaling * (
                            np.random.randint(0, 2, (self.output_dim, self.input_dim)) * 2.0 - 1.0)
                w_in = torch.from_numpy(w_in.astype(np.float32))
            else:
                w_in = self.input_scaling * np.random.choice(
                    np.append([0], self.input_set),
                    (self.output_dim, self.input_dim),
                    p=np.append([1.0 - self.sparsity],[self.sparsity / len(self.input_set)] * len(self.input_set))
                )
                w_in = torch.from_numpy(w_in.astype(np.float32))
        else:
            if callable(w_in):
                w_in = w_in(self.output_dim, self.input_dim)
        return w_in.requires_grad_(requires_grad=False)
    

    def _generate_wbias(self, w_bias):
        """
        Generate Wbias matrix
        :return:
        """
        if w_bias is None:
            w_bias = self.bias_scaling * (torch.rand(1, self.output_dim) * 2.0 - 1.0)
        else:
            if callable((w_bias)):
                w_bias = w_bias(self.output_dim)
        return w_bias.requires_grad_(requires_grad=False)
    

    def _generate_wfdb(self, w_fdb):
        """
        Generate Wfdb (feedback) matrix
        :return:
        """
        if w_fdb is None:
            if self.wfdb_sparsity is None:
                w_fdb = self.input_scaling * (
                        np.random.randint(0, 2, (self.output_dim, self.feedbacks_dim)) * 2.0 - 1.0)
                w_fdb = torch.from_numpy(w_fdb.astype(np.float32))
            else:
                w_fdb = self.input_scaling * np.random.choice(
                    np.append([0], self.input_set),
                    (self.output_dim, self.feedbacks_dim),
                    p=np.append([1.0 - self.wfdb_sparsity],[self.wfdb_sparsity / len(self.input_set)] * len(self.input_set))
                )
                w_fdb = torch.from_numpy(w_fdb.astype(np.float32))
        else:
            if callable(w_fdb):
                w_fdb = w_fdb(self.output_dim, self.feedbacks_dim)
        return w_fdb.requires_grad_(requires_grad=False)
    

    # Generate W matrix
    @staticmethod
    def generate_w(output_dim, w_sparsity=None):
        """
        Generate W matrix
        :param output_dim:
        :param w_sparsity:
        :return:
        """
        if w_sparsity is None:
            w = torch.rand(output_dim, output_dim) * 2.0 - 1.0
            return w
        else:
            w = np.random.choice(
                [0.0, 1.0],
                (output_dim, output_dim),
                p=[1.0 - w_sparsity, w_sparsity]
            )
            w[w == 1] = np.random.rand(len(w[w == 1])) * 2.0 - 1.0
            w = torch.from_numpy(w.astype(np.float32))
            return w

    # To sparse matrix
    @staticmethod
    def to_sparse(m):
        """
        To sparse matrix
        :param m:
        :return:
        """
        rows = torch.LongTensor()
        columns = torch.LongTensor()
        values = torch.FloatTensor()

        # For each row
        for i in range(m.shape[0]):
            # For each column
            for j in range(m.shape[1]):
                if m[i, j] != 0.0:
                    rows = torch.cat((rows, torch.LongTensor([i])), dim=0)
                    columns = torch.cat((columns, torch.LongTensor([j])), dim=0)
                    values = torch.cat((values, torch.FloatTensor([m[i, j]])), dim=0)
        indices = torch.cat((rows.unsqueeze(0), columns.unsqueeze(0)), dim=0)
        return torch.sparse.FloatTensor(indices, values)
    

    ## Forward Method
    def forward(self, u, hidden_x=None, y=None, w_out=None):
        """
        Forward
        :param u: Input signal
        :param y: Target output signal for teacher forcing
        :param w_out: Output weights for teacher forcing
        :return: Resulting hidden states
        """
        n_batches = int(u.size()[0])
        
        outputs = torch.zeros(n_batches, self.output_dim)
        outputs = outputs.cuda() if self.hidden.is_cuda else outputs
        if hidden_x is not None:
            self.hidden = hidden_x
        # For each batch
        for b in range(n_batches):
            # Reset hidden layer
            # self.reset_hidden()


            ut = u[b].clone() # Current input
            u_win = self.w_in.mv(ut.view(-1)) # Compute input layer
            x_w = self.w.mv(self.hidden) # Apply W to x

            # Feedback or not
            if self.feedbacks and self.training and y is not None:
                yt = y[b].view(-1) # Current target
                y_wfdb = self.w_fdb.mv(yt) # Compute feedback layer
                x = u_win + x_w + y_wfdb + self.w_bias
            elif self.feedbacks and not self.training and w_out is not None:
                bias_hidden = torch.cat([torch.ones(1), self.hidden], dim=0) # Add bias
                yt = w_out.t().mv(bias_hidden) # Compute past output
                if self.normalize_feedbacks:
                    yt -= torch.min(yt)
                    yt /= torch.max(yt) - torch.min(yt)
                    yt /= torch.sum(yt)
                y_wfdb = self.w_fdb.mv(yt)
                x = u_win + x_w + y_wfdb + self.w_bias
            else:
                x = u_win + x_w + self.w_bias
            x = self.nonlin_func(x) # Apply activation function
            self.hidden = x.view(self.output_dim)*(1-self.leaky_rate) + self.hidden*self.leaky_rate
            outputs[b] = x.view(self.output_dim)*(1-self.leaky_rate) + self.hidden*self.leaky_rate
        return outputs