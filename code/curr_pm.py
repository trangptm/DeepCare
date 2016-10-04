__author__ = 'ptmin'

from collections import OrderedDict
import numpy
import theano
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as tensor
import lstm_layer

SEED = 123
numpy.random.seed(SEED)

def load_options(
        dim_emb=10,
        dim_prj=20,
        dim_time=3,
        n_pred=3,
        lrate=0.01,

        L1_reg=0.00015,
        L2_reg=0.00025,

        max_len=30,
        batch_size=16,
        max_epochs=300
):
    options = locals().copy()
    return options

def init_top_params(options, params):
    params['V1'] = 1.0 * numpy.random.rand(options['dim_prj'], options['dim_emb']).astype(config.floatX)
    params['V2'] = 1.0 * numpy.random.rand(options['dim_emb'], options['n_pm']).astype(config.floatX)

    params['c1'] = numpy.zeros((options['dim_emb'],)).astype(config.floatX)
    params['c2'] = numpy.zeros((options['n_pm'],)).astype(config.floatX)

    return params

def random_lengths(seqs, adm,
                            duration=None, options=None, max_len=30, curr_time=106200):
    lengths = [0] * len(seqs)
    for i in range(len(seqs)):
        ids = seqs[i]
        pos = min(max_len, len(ids) - 1)
        if pos < 1: continue
        low = (pos + 1) / 2
        for turn in range(5):
            lengths[i] = numpy.random.randint(low, pos + 1)
            if adm['admit_time'][ids[lengths[i]-1] + 365.0 > adm['admit_time'][ids[lengths[i]]]]:
                break
    return lengths

def prepare_data(lens, seqs, adm,
                 duration = None, options=None, max_len=30, curr_time = 106200):

    pm_len = [len(pm) for pm in adm['pm']]
    lengths = [0] * len(seqs)
    for i, seq in enumerate(seqs):
        for adm_idx in seq:
            if pm_len[adm_idx] > 0:
                lengths[i] += 1

    n_samples = 0
    for l in lengths:
        if l > 1: n_samples += 1

    maxlen = numpy.max(lengths)
    x = numpy.zeros((maxlen, n_samples)).astype('int64')

    time = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    x_mask = numpy.zeros((maxlen, 2, n_samples)).astype(theano.config.floatX) #x_mask[:, 0]: diag, x_mask[:, 1]: pm

    med = numpy.ones((maxlen, n_samples)).astype(theano.config.floatX)
    y = numpy.zeros((maxlen, n_samples)).astype('int64')

    idx = 0
    for i, seq in enumerate(seqs):
        if lengths[i] < 2: continue

        j = 0
        for adm_idx in seq:
            if pm_len[adm_idx] > 0:
                x[j, idx] = adm_idx
                y[j, idx] = adm_idx
                j += 1
        x_mask[:lengths[i], 0, idx] = 1

        for j in range(lengths[i]):
            if j > 0: time[j, idx] = (adm['admit_time'][seq[j]] - adm['admit_time'][seq[j - 1]]) / 24.0

            if adm['method'][seq[j]] == 'e': med[j, idx] = 1
            else: med[j, idx] = 2

        idx += 1

    return x, x_mask, time, med, y

def prepare_train(seqs, adm, duration, options=None):
    lengths = random_lengths(seqs, adm, duration)
    x, x_mask, time, med, y = prepare_data(lengths, seqs, adm, duration)
    return x, x_mask, time, med, y

def build_model(shared_params, options, use_noise = None):
    trng = RandomStreams(SEED)
    # Used for dropout.
    if use_noise is None:
        use_noise = theano.shared(lstm_layer.numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.tensor3('x_mask', dtype=config.floatX)
    x_time = tensor.matrix('x_time', dtype=config.floatX)
    method = tensor.matrix('method', dtype=config.floatX)
    y = tensor.matrix('y', dtype='int64')

    adm_list = tensor.tensor3('adm_list', dtype='int64')
    adm_mask = tensor.tensor3('adm_mask', dtype=config.floatX)

    n_steps = x.shape[0]
    n_samples = x.shape[1]
    n_words = adm_list.shape[2]

    # compute mean vector for diagnosis & proc/medi
    for i in range(2):
        adm_words = adm_list[i][x.flatten()]
        word_mask = adm_mask[i][x.flatten()]
        if 'drin' in options['reg']:
            word_mask = lstm_layer.dropout_layer(word_mask, use_noise, trng, 0.8)

        emb_vec = shared_params['Wemb'][adm_words.flatten()].reshape([n_steps*n_samples,
                                                                      n_words,
                                                                      options['dim_emb']])
        mean_vec = (emb_vec * word_mask[:, :, None]).sum(axis=1)
        if options['embed'] == 'mean':
            mean_vec = mean_vec / word_mask.sum(axis=1)[:, None]
        elif options['embed'] == 'sum':
            mean_vec = mean_vec / tensor.sqrt(word_mask.sum(axis=1))[:, None]
        elif options['embed'] == 'max':
            mean_vec = (emb_vec * word_mask[:, :, None]).max(axis=1)
        elif options['embed'] == 'sqrt':
            mean_vec = mean_vec / tensor.sqrt(abs(mean_vec))

        emb_vec = mean_vec.reshape([n_steps, n_samples, options['dim_emb']])

        if 'drfeat' in options['reg']:
            emb_vec = lstm_layer.dropout_layer(emb_vec, use_noise, trng, 0.8)

        if i == 0: emb_dia = emb_vec
        else: emb_pm = emb_vec

    proj = lstm_layer.lstm_layer(shared_params, options, emb_dia, emb_pm, x_mask, x_time, method, 0)

    hid1 = tensor.dot(proj, shared_params['V1']).flatten().reshape([n_steps * n_samples, options['dim_emb']]) + shared_params['c1']
    #hid1 = tensor.nnet.sigmoid(hid1)

    hid2 = tensor.dot(hid1, shared_params['V2']) + shared_params['c2']
    prob = tensor.nnet.softmax(hid2).flatten()

    esp = 1e-8
    if prob.dtype == 'float16': esp = 1e-6
    pred = -tensor.log(prob + esp)

    sorted_idx = pred.reshape([n_steps, n_samples, options['n_pm']]).argsort(axis=2)
    f_pred = theano.function(inputs=[x, x_mask, x_time, method, adm_list, adm_mask], outputs=sorted_idx, name='f_pred')

    curr_pm = adm_list[1][y.flatten()] - options['n_diag']
    pm_mask = adm_mask[1][y.flatten()]
    pm = curr_pm + (tensor.arange(curr_pm.shape[0]) * options['n_pm'])[:, None]

    pm_pred = pred[pm.flatten()].reshape([curr_pm.shape[0], curr_pm.shape[1]])
    pm_pred = (pm_pred * pm_mask).sum(axis=1)
    pm_pred = pm_pred / pm_mask.sum(axis=1)

    pm_pred = (pm_pred.reshape([n_steps, n_samples]) * x_mask[:, 0]).sum(axis=0)
    pm_pred = pm_pred / x_mask[:, 0].sum(axis=0)

    cost = tensor.mean(pm_pred)

    if 'norm' in options['reg']:
        cost += options['L1_reg'] * lstm_layer.L1_reg(shared_params) + options['L2_reg'] * lstm_layer.L2_reg(shared_params)

    return x, x_mask, x_time, method, y, adm_list, adm_mask, f_pred, cost, use_noise

def init_best():
    return [0] * 500, [0] * 500

def update_best(best_valid, best_test, v_eval, t_eval):
    is_updated = 0
    if best_valid[0] < v_eval[0]:
        best_valid = v_eval
        best_test = t_eval
        is_updated = 1

    return best_valid, best_test, is_updated

def evaluation(pred, y, mask, adm, options):
    n_steps, n_samples = y.shape
    n_pred = options['n_pred']

    acc = [0] * n_pred
    total = 0
    for i in range(n_steps):
        for j in range(n_samples):
            if mask[i, 0, j] == 0: continue
            total += 1
            for t in range(n_pred):
                if pred[i, j, t] + options['n_diag'] in adm['pm'][y[i, j]]:
                    acc[t] += 1

    total_acc = 0
    for t in range(n_pred):
        total_acc += acc[t]
        acc[t] = (1.0 * total_acc) / (total * (t+1))

    return acc

def to_string(v_eval, t_eval):
    str = ''
    for val in t_eval:
        str += '%.4f\t' % (val)
    head = 'valid f-score: %.4f, test result: ' % (v_eval[0])
    return head + str