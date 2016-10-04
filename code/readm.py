__author__ = 'phtra'

import numpy
import theano
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as tensor
import lstm_layer

SEED = 123

def load_options(
        dim_emb=10,
        dim_prj=20,
        dim_hid=20,
        dim_time=3,
        dim_y=2,

        L1_reg=0.00015,
        L2_reg=0.00025,

        lrate=0.01,
        max_len=30,
        batch_size=16,
        max_epochs=200,
):
    options = locals().copy()
    return options

def init_top_params(options, params):
    low_val = -numpy.sqrt(24./(3*options['dim_prj'] + options['dim_hid']))
    high_val = -low_val
    params['U1'] = 1.0 * numpy.random.uniform(low=low_val, high=high_val,
                                              size=(3 * options['dim_prj'], options['dim_hid'])).astype(config.floatX)
    params['U2'] = 1.0 * numpy.random.rand(options['dim_hid'], options['dim_y']).astype(config.floatX)

    params['b1'] = numpy.zeros((options['dim_hid'],)).astype(config.floatX)
    params['b2'] = numpy.zeros((options['dim_y'],)).astype(config.floatX)

    return params

def random_lengths(seqs, adm,
                         duration=None, options=None, max_len=30, curr_time=106200):
    lengths = [0] * len(seqs)
    duration = duration * 24.0
    max_time = curr_time - duration #duration is 6 months/1 year, all the chosen times must be before max_time

    for i in range(len(seqs)):
        #seqs[i]: list of adm indeces for the patient i
        adm_ids = seqs[i]
        pos = -1

        # find the last admission which happened before max_time
        for j in range(len(adm_ids)): #seqs[i][j]: idx of the admission of patient i
            if adm['admit_time'][adm_ids[j]] <= max_time: pos = j

        pos = min(max_len - 1, pos)
        if pos >= 0: # pos >= 0 means there is some adm happened before max_time
            low = (pos + 1) / 2
            for turn in range(10):
                label = 0
                lengths[i] = numpy.random.randint(low, pos + 1) + 1 #randomly select a timeline
                time = adm['admit_time'][adm_ids[lengths[i] - 1]]

                for j in range(lengths[i], len(seqs[i])):
                    # if there is an emergency adm happening in the next 6 months -> lablel = 1
                    if (adm['method'][adm_ids[j]] == 'e') and (adm['admit_time'][adm_ids[j]] - time <= duration):
                        label = 1
                        break
                if (label == 1) and (lengths[i] > 1):
                    break
    return lengths

def prepare_data(lengths, seqs, adm,
                 duration = None, options=None, max_len=30, curr_time = 106200):

    duration *= 24.0
    pm_len = [len(pm) for pm in adm['pm']]
    n_samples = 0
    for l in lengths:
        if l > 1: n_samples += 1

    maxlen = numpy.max(lengths)
    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    time = numpy.zeros((2, maxlen, n_samples)).astype(theano.config.floatX)
    x_mask = numpy.zeros((maxlen, 5, n_samples)).astype(theano.config.floatX)
    med = numpy.ones((maxlen, n_samples)).astype(theano.config.floatX)
    y = numpy.zeros(n_samples).astype('int64')

    pooling_time = [duration/2.0, duration*1.0, duration * 2.0]

    idx = 0
    for i, seq in enumerate(seqs):
        if lengths[i] < 2: continue
        x[:lengths[i], idx] = seq[:lengths[i]]
        x_mask[:lengths[i], 0, idx] = 1

        for j in range(lengths[i], len(seq)):
            if (adm['method'][seq[j]] == 'e') and (adm['admit_time'][seq[j]] - adm['admit_time'][seq[lengths[i]-1]] <= duration):
                y[idx] = 1
        present_time = adm['admit_time'][seq[lengths[i] - 1]] + 100

        for j in range(lengths[i]):
            x_mask[j, 1, idx] = 1 if pm_len[seq[j]] > 0 else 0

            for id, pl in enumerate(pooling_time):
                if (present_time - adm['admit_time'][seq[j]]) <= pl:
                    x_mask[j, id + 2, idx] = 1

            if j > 0: time[0, j, idx] = (adm['admit_time'][seq[j]] - adm['admit_time'][seq[j - 1]]) / 24.0
            time[1, j, idx] = (present_time - adm['admit_time'][seq[j]]) / 24.0

            if adm['method'][seq[j]] == 'e': med[j, idx] = 1
            else: med[j, idx] = 2

        idx += 1

    return x, x_mask, time, med, y

def prepare_train(seqs, adm, duration=None, options=None):
    lengths = random_lengths(seqs, adm, duration)
    x, x_mask, time, med, y = prepare_data(lengths, seqs, adm, duration)
    return x, x_mask, time, med, y

def build_model(shared_params, options, use_noise = None):
    trng = RandomStreams(SEED)
    # Used for dropout.
    if use_noise is None:
        use_noise = theano.shared(lstm_layer.numpy_floatX(0.))

    x = tensor.matrix('x', dtype = 'int64')
    x_mask = tensor.tensor3('x_mask', dtype=config.floatX)
    x_time = tensor.tensor3('x_time', dtype=config.floatX)

    method = tensor.matrix('method', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

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

    proj = lstm_layer.lstm_layer(shared_params, options, emb_dia, emb_pm, x_mask, x_time[0], method)

    # weighted mean of hidden states - weighted funcion: 1/log(x_mask)
    weight = x_mask[:, 0] / (method + tensor.log(x_time[1]/30.0 + 1))# / tensor.log(x_time[1])
    weight0 = weight * x_mask[:, 3]
    weight1 = weight * x_mask[:, 4]
    weight2 = weight

    hidd_0 = tensor.sum(proj * weight0[:, :, None], axis=0) / tensor.sum(weight0, axis=0)[:, None]
    hidd_1 = tensor.sum(proj * weight1[:, :, None], axis=0) / tensor.sum(weight1, axis=0)[:, None]
    hidd_2 = tensor.sum(proj * weight2[:, :, None], axis=0) / tensor.sum(weight2, axis=0)[:, None]
    hidd = tensor.concatenate([hidd_0, hidd_1, hidd_2], axis=1)

    if 'drhid' in options['reg']:
         hidd = lstm_layer.dropout_layer(hidd, use_noise, trng, 0.9)

    # pool the hidden state to a neural network
    hid1 = tensor.dot(hidd, shared_params['U1']) + shared_params['b1']
    hid1 = tensor.nnet.sigmoid(hid1)

    if 'drhid' in options['reg']:
        hid1 = lstm_layer.dropout_layer(hid1, use_noise, trng, 0.5)

    pred = tensor.nnet.softmax(tensor.dot(hid1, shared_params['U2']) + shared_params['b2'])
    f_pred = theano.function(inputs = [x, x_mask, x_time, method, adm_list, adm_mask],
                             outputs = pred.argmax(axis=1), name = 'f_pred')

    esp = 1e-8
    if pred.dtype == 'float16': esp = 1e-6
    cost = -tensor.log(pred[tensor.arange(n_samples), y] + esp).mean()

    if 'norm' in options['reg']:
        cost += options['L1_reg'] * lstm_layer.L1_reg(shared_params) + options['L2_reg'] * lstm_layer.L2_reg(shared_params)

    return x, x_mask, x_time, method, y, adm_list, adm_mask, f_pred, cost, use_noise

def init_best():
    #[0]: acc, [1]: precision, [2]: recall, [3]: f-score
    best_valid = [0] * 4
    best_test = [0] * 4

    return best_valid, best_test

def update_best(best_valid, best_test, v_eval, t_eval):
    # using f-score
    is_updated = 0
    if v_eval[3] > best_valid[3]:
        best_valid = v_eval
        best_test = t_eval
        is_updated = 1

    return best_valid, best_test, is_updated

def evaluation(y_pred, y, mask, adm, options):
    tp = numpy.sum(y_pred * y)
    diff = numpy.abs(y_pred - y)
    wrong = 1.0 * diff.sum()
    acc = 1.0 - wrong / len(y)
    pre = 1.0 * tp / numpy.sum(y_pred)
    rec = 1.0 * tp / numpy.sum(y)
    f_score = 2 * pre * rec / (pre + rec)

    return acc, pre, rec, f_score

def to_string(v_eval, t_eval):
    str = 'valid f-score: %.4f, test result: %.4f\t%.4f\t%.4f\t%.4f' % (v_eval[3], t_eval[0], t_eval[1], t_eval[2], t_eval[3])
    return str

def prepare_data_long(lengths, seqs, adm,
                 duration = None, options=None, max_len=30, curr_time = 106200):

    duration *= 24.0
    pm_len = [len(pm) for pm in adm['pm']]
    n_samples = 0
    for l in lengths:
        if l > 9: n_samples += 1

    maxlen = numpy.max(lengths)
    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    time = numpy.zeros((2, maxlen, n_samples)).astype(theano.config.floatX)
    x_mask = numpy.zeros((maxlen, 5, n_samples)).astype(theano.config.floatX)
    med = numpy.ones((maxlen, n_samples)).astype(theano.config.floatX)
    y = numpy.zeros(n_samples).astype('int64')

    pooling_time = [duration/2.0, duration*1.0, duration * 2.0]

    idx = 0
    for i, seq in enumerate(seqs):
        if lengths[i] < 10: continue
        x[:lengths[i], idx] = seq[:lengths[i]]
        x_mask[:lengths[i], 0, idx] = 1

        for j in range(lengths[i], len(seq)):
            if (adm['method'][seq[j]] == 'e') and (adm['admit_time'][seq[j]] - adm['admit_time'][seq[lengths[i]-1]] <= duration):
                y[idx] = 1
        present_time = adm['admit_time'][seq[lengths[i] - 1]] + 100

        for j in range(lengths[i]):
            x_mask[j, 1, idx] = 1 if pm_len[seq[j]] > 0 else 0

            for id, pl in enumerate(pooling_time):
                if (present_time - adm['admit_time'][seq[j]]) <= pl:
                    x_mask[j, id + 2, idx] = 1

            if j > 0: time[0, j, idx] = (adm['admit_time'][seq[j]] - adm['admit_time'][seq[j - 1]]) / 24.0
            time[1, j, idx] = (present_time - adm['admit_time'][seq[j]]) / 24.0

            if adm['method'][seq[j]] == 'e': med[j, idx] = 1
            else: med[j, idx] = 2

        idx += 1

    return x, x_mask, time, med, y