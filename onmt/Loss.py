"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

import onmt
from collections import namedtuple


class LossComputeBase(nn.Module):
    """
    This is the loss criterion base class. Users can implement their own
    loss computation strategy by making subclass of this one.
    Users need to implement the compute_loss() method.
    We inherits from nn.Module to leverage the cuda behavior.
    """
    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[onmt.IO.PAD_WORD]

    def forward(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define the compute_loss().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs: additional info for computing loss.
        """
        # Need to simplify this interface.
        return self.compute_loss(batch, output, target, **kwargs)

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size):
        """
        Compute the loss in shards for efficiency.
        """
        batch_stats = onmt.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        gen_state = make_gen_state(output, batch, attns, range_,
                                   self.copy_attn)

        for shard in shards(gen_state, shard_size):
            loss, stats = self.compute_loss(batch, **shard)
            loss.div(batch.batch_size).backward()
            batch_stats.update(stats)

        return batch_stats

        def aux_consume_src(self, src, src_length):
            pass  # does nothing unless using NMT aux model

    def stats(self, loss, scores, target):
        """
        Compute and return a Statistics object.

        Args:
            loss(Tensor): the loss computed by the loss criterion.
            scores(Tensor): a sequence of predict output with scores.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum()
        return onmt.Statistics(loss[0], non_padding.sum(), num_correct)

    def bottle(self, v):
        return v.view(-1, v.size(2))

    def unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, generator, tgt_vocab):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)

        self.copy_attn = False
        weight = torch.ones(len(tgt_vocab))
        weight[self.padding_idx] = 0
        self.criterion = nn.NLLLoss(weight, size_average=False)

    def compute_loss(self, batch, output, target, **kwargs):
        """ See base class for args description. """
        scores = self.generator(self.bottle(output))
        scores_data = scores.data.clone()

        target = target.view(-1)
        target_data = target.data.clone()

        loss = self.criterion(scores, target)
        loss_data = loss.data.clone()

        stats = self.stats(loss_data, scores_data, target_data)

        return loss, stats

######################


class NMTKLDivNMTLossCompute(LossComputeBase):
    """
    NMT Loss Computation with KL divergence between LM distribution and NMT model distribution
    (see Rethinking the Inception Architecture for Computer Vision)

    In theory, setting smoothing_epsilon==0 should be same as NMTLossCompute.
    But for now I'm keeping all the bad copy-code that I can compare this
    implemementation directly to the un-mucked-with one.
    """
    def __init__(self, generator, tgt_vocab, smoothing_epsilon, aux_checkpoint):
        super(NMTKLDivNMTLossCompute, self).__init__(generator, tgt_vocab)
        self.copy_attn = False
        weight = torch.ones(len(tgt_vocab))
        weight[self.padding_idx] = 0

        # standard NLL loss term:
        self.criterion0 = nn.NLLLoss(weight, size_average=False)

        # ratio between normal cross entropy and LM cross entropy
        self.smoothing_epsilon = smoothing_epsilon

        #initial the aux model
        # The first argument of onmt.Translator just needs *.model and *.gpu
        OptModel = namedtuple('OptModel', ['model', 'gpu'])
        opt_model = OptModel(model=aux_checkpoint, gpu=0)  # TODO how do we know it is gpu 0 ???????????
        self.translator = onmt.Translator(opt_model, dict())

        # # ONLY USED FOR DEBUG
        self.debugLangModelNLL = nn.NLLLoss(weight, size_average=False)

        # print('foo')
        # for thing, param in zip(self.translator.model.state_dict(), self.translator.model.parameters()):
        #     print(thing, type(param.data), param.size(), hash(param), id(param))
        # print('end foo')

    #function that takes in the NMT source side, and updated the input state of the NMT model to use it
    def aux_consume_src(self, src, src_lengths):
        encStates, self.context = self.translator.model.encoder(src, src_lengths)
        self.decStates = self.translator.model.decoder.init_decoder_state(src, self.context, encStates)

        # print('src:', src.squeeze(2).data)
        # for sent in src.squeeze(2).transpose(0, 1).data:
        #     print([self.translator.fields["src"].vocab.itos[i] for i in sent])
        #     break

    def compute_loss(self, batch, output, target, **kwargs):
        """ See base class for args description. """

        # print('tgt:', target.data)
        # for sent in target.transpose(0, 1).data:
        #     print([self.translator.fields["tgt"].vocab.itos[i] for i in sent])
        #     break

        scores = self.generator(self.bottle(output))  # scores are log-probs
        scores_data = scores.data.clone()

        # target does not contain start tokens, so have to add them (else aux system is one timestep off)
        BOS_int = self.translator.fields["tgt"].vocab.stoi[onmt.IO.BOS_WORD]
        start_tokens = Variable(torch.zeros(1, output.size()[1]).type(torch.LongTensor) + BOS_int)
        start_tokens = start_tokens.cuda()
        target2 = torch.cat([start_tokens, target.unsqueeze(2)], dim=0)[:-1, :]  # :-1 -  just to keep size same
        # TODO: remove EOS ?? (must do per sent) Don't think it matters because there is a mask
        target = target.view(-1)
        target_data = target.data.clone()

        decOut, decStates, attn = self.translator.model.decoder(target2, self.context, self.decStates)
        aux_logprobs = torch.cat([self.translator.model.generator.forward(dec).unsqueeze(0) for dec in decOut])

        # print('Model predictions (given all previous gold words, not a true translation):')
        # import numpy as np
        # for sent in np.argmax(aux_logprobs.cpu().data.numpy(), axis=2).transpose():
        #     print([self.translator.fields["tgt"].vocab.itos[i] for i in sent])
        #     break

        aux_probs=torch.exp(aux_logprobs)
        # debug_NLL = self.debugLangModelNLL(aux_logprobs.view(scores.size()), target)
        # aux_stats = self.stats(loss=debug_NLL.data.clone(),
        #                        scores=aux_logprobs.view(scores_data.size()).data.clone(),
        #                        target=target_data)
        # print('output for aux stats:')
        # aux_stats.output(epoch=-1, batch=42, n_batches=42, start=42)

        loss0 = self.criterion0(scores, target)  # criterion0 = nn.NLLLoss(weight, size_average=False)

        # cross entropy with teacher probability distribution = dot teacher probs with model log probs
        # as in "Sequence-Level Knowledge Distillation" by Kim and Rush
        loss1 = - torch.mm(aux_probs.detach().view(1, -1), scores.view(-1, 1))
        loss = (1-self.smoothing_epsilon) * loss0 + self.smoothing_epsilon * loss1
        # print('normal loss=', loss0.data[0], 'aux loss=', loss1.data[0], 'combined loss=', loss.data[0], 'w/ lambda=', self.smoothing_epsilon)

        loss_data = loss0.data.clone()  # do stats on just the NLL Loss, not the regularization term

        stats = self.stats(loss_data, scores_data, target_data)
        # print('output for normal stats:')
        # stats.output(epoch=-2, batch=42, n_batches=42, start=42)

        return loss, stats



def make_gen_state(output, batch, attns, range_, copy_attn=None):
    """
    Create generator state for use in sharded loss computation.
    This needs to match compute_loss exactly.
    """
    if copy_attn and getattr(batch, 'alignment', None) is None:
        raise AssertionError("using -copy_attn you need to pass in "
                             "-dynamic_dict during preprocess stage.")

    return {"output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "copy_attn": attns.get("copy"),
            "align": None if not copy_attn
            else batch.alignment[range_[0] + 1: range_[1]],
            "coverage": attns.get("coverage")}


def filter_gen_state(state):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=True, volatile=False)
            yield k, v


def shards(state, shard_size, eval=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               make_gen_state(). The values for those keys are
               Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    yields:
        Each yielded shard is a dict.
    side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield state
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_gen_state(state))

        # Now, the iteration:
        # split_state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, torch.split(v, shard_size))
                             for k, v in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = ((state[k], v.grad.data) for k, v in non_none.items()
                     if isinstance(v, Variable) and v.grad is not None)
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
