# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:55:03 2015

"""
import numpy as np
import re

def load_vocabulary(fn):
    """
    """
    with open(fn) as f:
        vocabulary = f.read().splitlines()
        f.close()
    return vocabulary


def extract_vocabulary(fn):
    vocabulary = set()
    with open(fn) as f:
        for line in f:
            word_list = re.findall('\([^(^)]*\)', line)
            for word in word_list:
                vocabulary.add(word[3:-1])
    f.close()
    return list(vocabulary)

    
def save_vocabulary(fn_out, vocabulary):
    f = open(fn_out, 'w')
    for word in vocabulary:
        f.write('%s\n' % word)
    f.close()
                                                                                                     

def get_proto_tree(f):
    '''
    f        -- open file, or file name
    fn_vocab -- e.g.: 'vocabulary.txt', text file with all the words listed (V).
    params
    

    
    Tree in PTB format
    Convert text from the PTB to a tree.
    Examle from figure 4 in paper:
    (1 (2 'not') (4 (2 'very') (3 'good')))
    Graphially:
             p2(1)
            /    \
          /       p1(4)
        /        /     \
    'not'(2) 'very'(2) 'good'(3)
    '''

    if type(f) is str:
        f = open(f)
    
    line = f.readline()
    if len(line) < 1:
        return None

    line = line.strip('\n')
    nodes = re.findall('\([0-4] ', line)
    # Put commmas after all numbers        
    scores = np.unique(nodes)
    N = len(nodes)
    for score in scores:
        line = line.replace(score, '%s,' % score)
    leaves = re.findall('\([^(^)]*\)', line)
    # Put " around strings
    for leaf in leaves:
        word = leaf[4:-1]
        mod_leaf = '%s"%s"%s' % (leaf[:4], leaf[4:-1], leaf[-1])
        line = line.replace(leaf, mod_leaf)
    # Put commas between neighboring parantheses
    line = line.replace(') (', '), (')
    line = line.replace(')', ']')  # make it a list
    line = line.replace('(', '[')

    return eval(line), N


def read_proto_trees(fn_data):
    """
    """
    
    f = open(fn_data)
    
    trees = []
    Ns = []
    while True:
        tree = get_proto_tree(f)
        if tree is None:
            break
        trees.append(tree[0])
        Ns.append(tree[1])

    f.close()
    return trees, Ns
    