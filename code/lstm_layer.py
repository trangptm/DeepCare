__author__ = 'phtra'

from collections import OrderedDict

import numpy
import theano
import theano.tensor as tensor
from theano import config

SEED = 123

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

def L1_reg(params):
    s = theano.shared(0)
    for key, val in params.iteritems():
        s += abs(val).sum()
    return s

def L2_reg(params):
    s = theano.shared(0)
    for key, val in params.iteritems():
        s += (val ** 2).sum()

    return s

def lstm_layer(shared_params, options, emb_dia, emb_pm, x_mask, time, method, use_pm = 1):
    n_steps = emb_dia.shape[0]
    if emb_dia.ndim == 3:
        n_samples = emb_dia.shape[1]
    else: n_samples = 1

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n+1) * dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(x_, xf_, m_, time_, method_, h_, c_, pm_):
        preact = tensor.dot(x_, shared_params['lstm_W']) \
                + tensor.dot(h_, shared_params['lstm_U']) + shared_params['lstm_b']

        pm = m_[1, :, None] * xf_

        pre_f = tensor.dot(pm_, shared_params['lstm_Pf'])
        pre_o = tensor.dot(pm, shared_params['lstm_Po'])
        time = tensor.concatenate([[time_/60.0], [(time_/180.0) ** 2], [(time_/365.0) ** 3]])
        time = tensor.transpose(time)
        pre_t = tensor.dot(time, shared_params['lstm_Z'])

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_prj'])) * (1.0 / method_[:, None])
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_prj']) + pre_f + pre_t)
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_prj']) + pre_o)
        c = tensor.tanh(_slice(preact, 3, options['dim_prj']))

        c = f * c_ + i * c
        c = m_[0, :, None] * c + (1.0 - m_[0])[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[0, :, None] * h + (1.0 - m_[0])[:, None] * h_

        if use_pm == 0:
            pm = xf_
        return h, c, pm

    dim_prj = options['dim_prj']
    dim_emb = options['dim_emb']
    rval, updates = theano.scan(_step,
                                sequences=[emb_dia, emb_pm, x_mask, time, method],
                                outputs_info=[tensor.alloc(numpy_floatX(0.), n_samples, dim_prj),
                                              tensor.alloc(numpy_floatX(0.), n_samples, dim_prj),
                                              tensor.alloc(numpy_floatX(0.), n_samples, dim_emb)],
                                name='lstm_layer', n_steps=n_steps)

    return rval[0]