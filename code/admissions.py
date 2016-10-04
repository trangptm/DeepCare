__author__ = 'ptmin'

import cPickle
import numpy
import theano
from theano import config
import theano.tensor as tensor

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def dropout_layer(state_before, use_noise, trng, rate):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=rate, n=1,
                                        dtype=state_before.dtype)),
                         state_before * rate)
    return proj

def load_dict(dict_file = 'dict.pkl'):
    f = open(dict_file, 'rb')
    diag_dict, proc_dict, medi_dict = cPickle.load(f)

    return diag_dict, proc_dict, medi_dict

def prepare_adm(diag_set, pm_set):
    n_adm = len(diag_set)
    lengths = [ [len(diag) for diag in diag_set], [len(pm) for pm in pm_set] ]
    max_len = numpy.max(lengths)

    adm_list = numpy.zeros((2, n_adm, max_len)).astype('int64')
    adm_mask = numpy.zeros((2, n_adm, max_len)).astype(theano.config.floatX)

    for idx, diag in enumerate(diag_set):
        adm_list[0, idx, :lengths[0][idx]] = diag[:lengths[0][idx]]
        adm_mask[0, idx, :lengths[0][idx]] = 1

    for idx, pm in enumerate(pm_set):
        if lengths[1][idx] == 0:
            pm = [0]
            lengths[1][idx] = 1
        adm_list[1, idx, :lengths[1][idx]] = pm[:lengths[1][idx]]
        adm_mask[1, idx, :lengths[1][idx]] = 1

    return adm_list, adm_mask