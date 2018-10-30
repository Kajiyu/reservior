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
        self, input_dim, output_dim, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0, w=None, w_in=None,
        w_bias=None, w_fdb=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None,
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
        self.nonlin_func = nonlin_func
        self.feedbacks = feedbacks
        self.feedbacks_dim = feedbacks_dim
        self.wfdb_sparsity = wfdb_sparsity
        self.normalize_feedbacks = normalize_feedbacks
        # Init hidden state
        self.register_buffer('hidden', self.init_hidden())
        # Initialize input weights
        self.register_buffer('w_in', self._generate_win(w_in))
        # Initialize reservoir weights randomly
        self.register_buffer('w', self._generate_w(w))
        # Initialize bias
        self.register_buffer('w_bias', self._generate_wbias(w_bias))
        # Initialize feedbacks weights randomly
        if feedbacks:
            self.register_buffer('w_fdb', self._generate_wfdb(w_fdb))
    

    def init_hidden(self):
        return Variable(torch.zeros(self.output_dim), requires_grad=False)

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
        return Variable(w, requires_grad=False)
    

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
        return Variable(w_in, requires_grad=False)
    

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
        return Variable(w_bias, requires_grad=False)
    

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
        return Variable(w_fdb, requires_grad=False)
    

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
    def forward(self, u, y=None, w_out=None):
        """
        Forward
        :param u: Input signal
        :param y: Target output signal for teacher forcing
        :param w_out: Output weights for teacher forcing
        :return: Resulting hidden states
        """
        time_length = int(u.size()[1])
        n_batches = int(u.size()[0])
        
        outputs = Variable(torch.zeros(n_batches, time_length, self.output_dim))
        outputs = outputs.cuda() if self.hidden.is_cuda else outputs

        # For each batch
        for b in range(n_batches):
            # Reset hidden layer
            # self.reset_hidden()

            # For each steps
            for t in range(time_length):
                ut = u[b, t] # Current input
                # print(ut.view(-1).size())
                u_win = self.w_in.mv(ut.view(-1)) # Compute input layer
                x_w = self.w.mv(self.hidden) # Apply W to x

                # Feedback or not
                if self.feedbacks and self.training and y is not None:
                    yt = y[b, t].view(-1) # Current target
                    y_wfdb = self.w_fdb.mv(yt) # Compute feedback layer
                    x = u_win + x_w + y_wfdb + self.w_bias
                elif self.feedbacks and not self.training and w_out is not None:
                    bias_hidden = torch.cat((Variable(torch.ones(1)), self.hidden), dim=0) # Add bias
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
                self.hidden.data = x.view(self.output_dim).data
                outputs[b, t] = self.hidden
        return outputs


class ESN(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0,
        w=None, w_in=None, w_bias=None, w_fdb=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None,
        nonlin_func=torch.tanh, feedbacks=False, with_bias=True, wfdb_sparsity=None, normalize_feedbacks=False, skip_output=False):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param hidden_dim: Hidden layer dimension
        :param output_dim: Reservoir size
        :param spectral_radius: Reservoir's spectral radius
        :param bias_scaling: Scaling of the bias, a constant input to each neuron (default: 0, no bias)
        :param input_scaling: Scaling of the input weight matrix, default 1.
        :param w: Internation weights matrix
        :param w_in: Input-reservoir weights matrix
        :param w_bias: Bias weights matrix
        :param w_fdb: Feedback weights matrix
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func: Reservoir's activation function (tanh, sig, relu)
        :param learning_algo: Which learning algorithm to use (inv, LU, grad)
        """
        super(ESN, self).__init__()
        self.output_dim = output_dim
        self.feedbacks = feedbacks
        self.with_bias = with_bias
        self.normalize_feedbacks = normalize_feedbacks

        self.esn_cell = ESNCell(
            input_dim, hidden_dim, spectral_radius, bias_scaling, input_scaling, w, w_in,
            w_bias, w_fdb, sparsity, input_set, w_sparsity, nonlin_func, feedbacks, output_dim,
            wfdb_sparsity, normalize_feedbacks)
        self.skip_output = skip_output
        if self.skip_output is not True: 
            self.output = nn.Linear(hidden_dim, output_dim)
    

    @property
    def hidden(self):
        return self.esn_cell.hidden
    
    
    @property
    def w(self):
        return self.esn_cell.w

    
    @property
    def w_in(self):
        return self.esn_cell.w_in
    
    
    def reset(self):
        self.output.reset()
    
    
    def get_w_out(self):
        return self.output.w_out
    
    
    def set_w(self, w):
        self.esn_cell.w = w
    
    
    # Forward
    def forward(self, u, y=None):
        """
        Forward
        :param u: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        # Compute hidden states
        if self.feedbacks and self.training:
            hidden_states = self.esn_cell(u, y)
        elif self.feedbacks and not self.training:
            hidden_states = self.esn_cell(u, w_out=self.output.w_out)
        else:
            hidden_states = self.esn_cell(u)
        
        if self.skip_output:
            out_vec = hidden_states
        else:
            out_vec = self.output(hidden_states)
        # out_vec = F.sigmoid(out_vec)
        return out_vec
    
    
    # Reset hidden layer
    def reset_hidden(self):
        self.esn_cell.reset_hidden()
    
    
    # Get W's spectral radius
    def get_spectral_radius(self):
        return self.esn_cell.get_spectral_raduis()