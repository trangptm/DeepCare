__author__ = 'phtra'

# joint tasts: readm & next_diag, high_risk & next_diag,
#              readm & curr_pm,   high_risk & curr_pm
# task1-task2

import numpy
import theano
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as tensor
import lstm_layer

import readm
import high_risk
import next_diag
import curr_pm

SEED = 123

tasks = {'readm': readm, 'high_risk': high_risk,
         'next_diag': next_diag, 'curr_pm': curr_pm}

def load_options(
        dim_emb=30,
        dim_prj=40,
        dim_hid=50,
        dim_time=3,
        dim_y=2,
        n_pred=3,

        L1_reg=0.0001,
        L2_reg=0.00015,

        lrate=0.01,
        max_len=30,
        batch_size=16,
        max_epochs=200,
):
    options = locals().copy()
    return options

def init_top_params(options, params):
    task1, task2 = options['task'].split('-')
    params = tasks[task1].init_top_params(options, params)
    params = tasks[task2].init_top_params(options, params)

    return params

def random_lengths(seqs, adm,
                         duration=None, options=None, max_len=30, curr_time=106200):
    task1, task2 = options['task'].split('-')
    lengths = tasks[task1].random_lengths(seqs, adm, duration)
    return lengths

def prepare_data(lengths, seqs, adm,
                 duration = None, options=None, max_len=30, curr_time = 106200):
    task1, task2 = options['task'].split('-')
    x1, x_mask1, time1, med1, y1 = tasks[task1].prepare_data(lengths, seqs, adm, duration)
    x2, x_mask2, time2, med2, y2 = tasks[task2].prepare_data(lengths, seqs, adm, duration)

    return [x1, x2], [x_mask1, x_mask2], [time1, time2], [med1, med2], [y1, y2]


def prepare_train(seqs, adm, duration=None, options=None):
    lengths = random_lengths(seqs, adm, duration, options)
    return prepare_data(lengths, seqs, adm, duration, options)

def build_model(shared_params, options):
    task1, task2 = options['task'].split('-')
    (x1, x_mask1, x_time1, method1, y1, adm_list1, adm_mask1, f_pred1, cost1, use_noise) = tasks[task1].build_model(shared_params, options)
    (x2, x_mask2, x_time2, method2, y2, adm_list2, adm_mask2, f_pred2, cost2, use_noise) = tasks[task2].build_model(shared_params, options)

    cost = cost1 + cost2

    return x1, x_mask1, x_time1, method1, y1, adm_list1, adm_mask1, f_pred1, \
           x2, x_mask2, x_time2, method2, y2, adm_list2, adm_mask2, f_pred2, cost, use_noise

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

def evaluation(y_pred, y, mask, adm, model_options):
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