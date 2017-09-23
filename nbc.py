# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 18:58:09 2017

@author: Parag Guruji
@email: pguruji@purdue.edu
"""

from sys import stdout
from os import path, makedirs
from codecs import open as codecs_open
from operator import itemgetter
from re import sub as re_sub
from random import shuffle, sample
from functools import reduce
from math import sqrt
from csv import DictWriter
from numpy import loadtxt as np_loadtxt
from matplotlib import pyplot


ALPHA = 1
CLASS_LABELS = [0, 1]
DEFAULT_STOPWORD_COUNT = 100
TOP_FEATURES_TO_PRINT = 10
DEFAULT_TRAINING_PERCENT = 50
DEFAULT_FEATURE_COUNT = 500


def get_feature_values(label):
    """Returns a list of all possible values a feature can take
    """
    return [0, 1]


def generate_features(labels):
    """Enumerate all features in the form of dict with each key being a \
        feature number and each resp value being a dict of the form: \
        {'label': <feature_label>, 'values': [<Value1>, <Value2>, ...]}
    """
    return dict([(i, {'label': labels[i],
                      'values': get_feature_values(labels[i])})
                 for i in xrange(len(labels))])


def read_tsv_file(file_name):
    """reads the given tab separated data file. If a line has only review text\
        (missing review_id and/or class_label, then appends it to the text of \
        previous valid review)
    """
    data = {}
    last_id = -1
    infp = codecs_open(file_name, "r", encoding='utf-8')
    for line in infp:
        line = [l.strip() for l in line.split('\t')]
        if len(line) != 3:
            if last_id >= 0:
                data[last_id]['review'] += " " + line[0].strip()
            else:
                raise ValueError
        else:
            data[int(line[0])] = {'id': int(line[0]),
                                  'class': int(line[1]),
                                  'review': line[2]}
            last_id = int(line[0])
    return data


def prepare_train_test_sets(whole_data_file, training_percent):
    """Randomly samples <training_percent>% of records from <whole_data_file> \
        as training set and rest as testing set, and returns both as tuple
    """
    whole_data = read_tsv_file(whole_data_file)
    size = int(training_percent * len(whole_data) / 100)
    random_ids = whole_data.keys()
    shuffle(random_ids)
    training_set = dict([(i, whole_data.pop(i)) for i in random_ids[:size]])
    return training_set, whole_data


def preprocess(data_set):
    """Preprocesses each review text in given dataset as following:
        1. Convert everything to lowercase
        2. Remove all characters except alphanumeric characters and whitespaces
        3. Split words on whitespaces
        4. Keep unique words of review in a set, add it to the record dict as \
            a value with key 'bow', discard original text entry from record
    """
    for i in data_set.keys():
        data_set[i]['bow'] = \
            set(re_sub(r'[^\w'+' '+']',
                       '',
                       data_set[i].pop('review').lower()).split())
    return data_set


def build_ordered_vocabulary(preprocessed_training_set):
    """Returns a list of unique words sorted in descending order of their \
        frequency. (frequency: no. of records containing that word)
        Tie Resolution: words with equal frequency are randomly shuffled \
            within the range of ranks their respective frequency had.
    """
    vocab = {}
    ordered_vocab = []
    for i in preprocessed_training_set:
        for w in preprocessed_training_set[i]['bow']:
            vocab[w] = 1 if w not in vocab else vocab[w] + 1

    count_words_pairs = sorted([(v, [k for k in vocab if vocab[k] == v])
                                for v in set(vocab.values())],
                               key=itemgetter(0),
                               reverse=True)

    for pair in count_words_pairs:
        ordered_vocab.extend(sample(pair[1], len(pair[1])))
    return ordered_vocab


def assign_feature_value(label, values, data):
    """Assigns binary value [0, 1] to the feature:
        if <label> (word) is present in set <data> (bag of words of a review):
            return 1
        else return 0
        if assigned value is not in <values> (allowed values for feature):
            raises ValueError
    """
    val = int(label in data)
    if val in values:
        return val
    else:
        raise ValueError


def build_feature_vector(bag, features):
    """Generates and returns the feature vector of dimensions <features> \
        for given data <bag>
    """
    return [assign_feature_value(features[i]['label'],
                                 features[i]['values'],
                                 bag) for i in features]


def build_feature_matrix(preprocessed_training_set, features):
    """Returns 2D array of size eqal to that of <preprocessed_training_data> \
        where each element is an array of size 2 + size of <features> and \
        format: [<feature_vector>, class_label, review_id]
    """
    return [build_feature_vector(preprocessed_training_set[i]['bow'],
                                 features) +
            [preprocessed_training_set[i]['class'],
             preprocessed_training_set[i]['id']]
            for i in preprocessed_training_set]


def compute_baseline_params(feature_matrix):
    """Computes baseline parameter PRIOR
    """
    params = {'prior': {}}
    prior_numerator = {}
    prior_denominator = float(len(feature_matrix))

    for k in CLASS_LABELS:
        prior_numerator[k] = len([feature_vector
                                  for feature_vector in feature_matrix
                                  if feature_vector[-2] == k])
        params['prior'][k] = prior_numerator[k] / prior_denominator
    return params


def baseline_classify(preprocessed_testing_set, baseline_params):
    """Returns result matrix of applying baseline classification to data
    """
    result_set = {}
    ids = preprocessed_testing_set.keys()
    for r in ids:
        result_set[r] = {'id': preprocessed_testing_set[r]['id'],
                         'class': preprocessed_testing_set[r]['class']}
        result_set[r]['probabilities'] = \
            dict([(k, baseline_params['prior'][k]) for k in CLASS_LABELS])
        result_set[r]['result'] = \
            max(result_set[r]['probabilities'].iteritems(),
                key=itemgetter(1))[0]
    return result_set


def laplace_smoothing(numerator, denominator, d=2):
    """Returns a Laplace smoothed value for given numerator & denominator terms
    """
    return float(numerator + ALPHA) / float(denominator + ALPHA*d)


def compute_nbc_params(feature_matrix, features):
    """Computes nbc parameters: PRIOR and CPDs
    """
    params = {'features': features}
    params['prior'] = {}
    prior_numerator = {}
    prior_denominator = float(len(feature_matrix))
    for k in CLASS_LABELS:
        prior_numerator[k] = len([feature_vector
                                  for feature_vector in feature_matrix
                                  if feature_vector[-2] == k])
        params['prior'][k] = prior_numerator[k] / prior_denominator

    params['cpd'] = {}
    for i in features:
        params['cpd'][i] = {}
        for j in features[i]['values']:
            params['cpd'][i][j] = {}
            for k in CLASS_LABELS:
                """cpd(Xi=j | Y=k)
                """
                params['cpd'][i][j][k] = \
                    laplace_smoothing(len([feature_vector for feature_vector
                                           in feature_matrix
                                           if feature_vector[i] == j and
                                           feature_vector[-2] == k]),
                                      prior_numerator[k])
    return params


def nbc_classify(preprocessed_testing_set, nbc_params):
    """Returns result matrix generated by applying the NB classification to \
        test data
    """
    result_set = {}
    ids = preprocessed_testing_set.keys()
    for r in ids:
        result_set[r] = {'id': preprocessed_testing_set[r]['id'],
                         'class': preprocessed_testing_set[r]['class']}
        result_set[r]['probabilities'] = {}
        feature_vector = \
            build_feature_vector(preprocessed_testing_set[r]['bow'],
                                 nbc_params['features'])
        for k in CLASS_LABELS:
            result_set[r]['probabilities'][k] = \
                reduce((lambda a, b: a * b),
                       [nbc_params['cpd'][i][feature_vector[i]][k]
                        for i in nbc_params['cpd']])
        result_set[r]['result'] = \
            max(result_set[r]['probabilities'].iteritems(),
                key=itemgetter(1))[0]
    return result_set


def evaluate_zero_one_loss(result_matrix):
    """Returns zero-one loss from given result_matrix
    """
    return sum([int(result_matrix[r]['class'] != result_matrix[r]['result'])
                for r in result_matrix]) / float(len(result_matrix))


def main(**kwargs):
    """Controller function for step-wise execution of learning and application\
        of baseline as well as NBC model

        :Returns: {'baseline': baseline_performance, 'nbc': nbc_performance}
        :kwargs:
            percent:    percent of whole data to be used for training
            train_set:  file-path of training set (when percent is absent \
                        --------------------------OR--------------------------\
                        file-path of whole dataset (when percent is present')
            test_set:   filenames of testing dataset
                        default=DEFAULT_TRAINING_PERCENT
            feature_count:  number of features
                            default=DEFAULT_FEATURE_COUNT
            stopword_count: number of most frequent words to be considered \
                            as stopwords
                            default=DEFAULT_STOPWORD_COUNT
            print:  Flag: when True, prints results for Q1 & Q2 to stdout
    """
    # when train and test data are given in the same file
    if 'percent' in kwargs and kwargs['percent']:
        training_set, testing_set = \
            prepare_train_test_sets(kwargs['train_set'],
                                    kwargs['percent'])
    # when separate train and test data files are given
    else:
        training_set = read_tsv_file(kwargs['train_set'])
        testing_set = read_tsv_file(kwargs['test_set'])

    preprocessed_training_set = preprocess(training_set)

    ranked_vocabulary = build_ordered_vocabulary(preprocessed_training_set)

    features = generate_features(
                ranked_vocabulary[DEFAULT_STOPWORD_COUNT:
                                  DEFAULT_STOPWORD_COUNT +
                                  kwargs['feature_count']])
    # print for Q1 & Q2
    if 'print' in kwargs and kwargs['print']:
        for i in xrange(TOP_FEATURES_TO_PRINT):
            print "WORD" + str(i+1) + " " + features[i]['label']

    feature_matrix = build_feature_matrix(preprocessed_training_set, features)

    baseline_params = compute_baseline_params(feature_matrix)
    nbc_params = compute_nbc_params(feature_matrix, features)

    preprocessed_testing_set = preprocess(testing_set)

    baseline_results = baseline_classify(preprocessed_testing_set,
                                         baseline_params)
    baseline_performance = evaluate_zero_one_loss(baseline_results)
    nbc_results = nbc_classify(preprocessed_testing_set, nbc_params)
    nbc_performance = evaluate_zero_one_loss(nbc_results)
    if 'print' in kwargs and kwargs['print']:
        print "ZERO-ONE-LOSS", nbc_performance
    return {'baseline': baseline_performance,
            'nbc': nbc_performance}


def run_experiment(arguments, repeat):
    """Runs an experiment with configuration specified by arguments repeat no.\
        of times with new randomly generated sample for each trial

        :Returns: {'trials': <dict of trial results>,
                   'mean': {'baseline': baseline_mean, 'nbc': nbc_mean},
                   'sigma': {'baseline': baseline_sigma, 'nbc': nbc_sigma}}
    """
    output = {'trials': {},
              'mean': {'baseline': 0.0, 'nbc': 0.0},
              'sigma': {}}
    stdout.write("\t\tTRIAL: ")
    stdout.flush()
    for i in range(1, repeat + 1):
        stdout.write(' ' + str(i))
        stdout.flush()
        output['trials'][i] = main(**arguments)
        output['mean']['baseline'] += output['trials'][i]['baseline']
        output['mean']['nbc'] += output['trials'][i]['nbc']
    stdout.write('\n')
    stdout.flush()
    for model in ['baseline', 'nbc']:
        output['mean'][model] /= float(repeat)
        output['sigma'][model] = \
            sqrt(
                sum(
                    [(output['trials'][i][model] -
                      output['mean'][model]) ** 2
                     for i in range(1, repeat + 1)]
                ) / float(repeat))
    return output


def measure_param(param,
                  param_label,
                  response_label,
                  param_values,
                  repeat,
                  data_path,
                  output_path,
                  xlim_margin=0):
    """Run multiple experiments by varying one <param> and plot graph for \
        comparing baseline with NBC
    """
    columns = [param_label,
               'baseline_mean',
               'baseline_sigma',
               'nbc_mean',
               'nbc_sigma']
    arguments = {'train_set': data_path,
                 'stopword_count': DEFAULT_STOPWORD_COUNT,
                 'feature_count': DEFAULT_FEATURE_COUNT,
                 'percent': DEFAULT_TRAINING_PERCENT,
                 'print': False}
    csv_rows = []
    for p in param_values:
        stdout.write("\n\t" + param_label + " = " + str(p) + '\n')
        stdout.flush()
        arguments[param] = p
        result = run_experiment(arguments, repeat)
        csv_rows.append({param_label: p,
                         'baseline_mean': result['mean']['baseline'],
                         'baseline_sigma': result['sigma']['baseline'],
                         'nbc_mean': result['mean']['nbc'],
                         'nbc_sigma': result['sigma']['nbc']})
        stdout.write("\tSUMMARY:\t" + "\t".join([c + "=" + str(csv_rows[-1][c])
                                                 for c in columns]) + '\n')
        stdout.flush()

    if not path.isdir('output'):
        makedirs('output')
    with open(output_path + '.csv', 'wb') as output_file:
        dict_writer = DictWriter(output_file, columns)
        dict_writer.writeheader()
        dict_writer.writerows(csv_rows)
    data = np_loadtxt([','.join([str(d[c]) for c in columns])
                       for d in csv_rows],
                      delimiter=",",
                      skiprows=0)
    pyplot.errorbar(x=data[:, 0],
                    y=data[:, 1],
                    yerr=data[:, 2],
                    color='blue',
                    label='baseline',
                    marker='o')

    pyplot.errorbar(x=data[:, 0],
                    y=data[:, 3],
                    yerr=data[:, 4],
                    color='green',
                    label='nbc',
                    marker='o')
    pyplot.xlabel(param_label)
    pyplot.ylabel(response_label)
    pyplot.title('CS 573 Data Mining HW-2: Naive Bayes Text Classifier\n\
                 By: Parag Guruji, pguruji@purdue.edu\n mean ' +
                 response_label + ' vs ' +
                 param_label + ' with standard deviation on error-bars\n',
                 loc='center')
    pyplot.xlim(pyplot.xlim()[0],
                pyplot.xlim()[1] + xlim_margin)
    pyplot.legend(loc='upper center', title='Legend')
    pyplot.savefig(output_path + '.png', bbox_inches='tight')
    pyplot.show()


def questions_3_and_4(data_path=''):
    """Stepwise execution of experiments for Q3 & Q4
        Generates output files (.csv and .png) in output subdirectory of \
        current working directory.
        configuration is hardcoded for Q3 & Q4.
    """
    if not data_path:
        return
    stdout.write("\n\n" + "*" * 80 + "\nExperiments for Que-3:\n" +
                 "*" * 80 + '\n')
    measure_param(param='percent',
                  param_label='training-set size (in % whole data)',
                  response_label='zero-one loss',
                  param_values=[1, 5, 10, 20, 50, 90],
                  repeat=10,
                  data_path=data_path,
                  output_path=path.join('output', 'question3'),
                  xlim_margin=10)
    stdout.write("\n\n" + "*" * 80 + "\nExperiments for Que-4:\n" + "*" * 80 +
                 '\n')
    measure_param(param='feature_count',
                  param_label='feature_count',
                  response_label='zero-one loss',
                  param_values=[10, 50, 250, 500, 1000, 4000],
                  repeat=10,
                  data_path=data_path,
                  output_path=path.join('output', 'question4'),
                  xlim_margin=40)


if __name__ == "__main__":
    """Process commandline arguments and make calls to appropriate functions
    """
    import argparse
    parser = \
        argparse.ArgumentParser(
                    description='CS 573 Data Mining HW2 NBC Implementation',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('train_set',
                        help='file-path of training set (when next positional \
                        argument is a string specifying file-path of test_set \
                        --------------------------OR--------------------------\
                        file-path of whole yelp dataset (when next positional \
                        argument is an integer specifying percent of whole \
                        data to be used for training).')
    parser.add_argument('test_set_or_percent',
                        nargs='?',
                        default=DEFAULT_TRAINING_PERCENT,
                        help='If string: filenames of testing dataset. . . . .\
                        . . .\
                        If integer: percent of data to be used for training')
    parser.add_argument('-f', '--feature_count',
                        metavar='feature_count',
                        type=int,
                        default=DEFAULT_FEATURE_COUNT,
                        help='number of features')
    parser.add_argument('-s', '--stopword_count',
                        metavar='stopword_count',
                        type=int,
                        default=DEFAULT_STOPWORD_COUNT,
                        help='number of most frequent words to be considered \
                        as stopwords')
    parser.add_argument('-e', '--evaluation',
                        metavar='evaluation',
                        default=None,
                        help="file-path of whole data set to be used for \
                        performance evaluation as specified in Q3 & Q4. The \
                        output files are stored in a subdirectory of current \
                        working directory named output (created if doesn't \
                        exist)")
    args = parser.parse_args()
    arguments = {'train_set': args.train_set,
                 'feature_count': args.feature_count,
                 'stopword_count': args.stopword_count,
                 'print': True}
    try:
        percent = int(args.test_set_or_percent)
        arguments['percent'] = percent
    except ValueError:
        arguments['test_set'] = args.test_set_or_percent
    main(**arguments)
    stdout.flush()
    questions_3_and_4(args.evaluation)
