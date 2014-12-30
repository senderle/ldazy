#!/usr/bin/python

# ldazy.py -- A toy implementation of an LDA gibbs sampler.
#             Totally unoptimized; using pypy is recommended!
#
#    Copyright 2014   by Jonathan Scott Enderle
#

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import sys
import itertools
import os
import re
import random
import collections
import heapq
import math
import argparse


def fprint_c(item, f, delim=' '):
    f.write(str(item))
    f.write(delim)

def fprint(item, f):
    f.write(str(item))
    f.write('\n')

class Corpus(object):
    def __init__(self, path, stopword_path=''):
        self._stopwords = self._load_stopwords(stopword_path)

        self._word_names = []
        self._wordname_to_index = {}
        self._chunk_names = []
        self._chunkname_to_index = {}
        self._chunk_tokens = []
        
        self._load_chunk_data(path)

        self._corpus_words = None
        self._corpus_chunks = None

    @property
    def n_chunks(self):
        return len(self._chunk_names)

    @property
    def vocab_size(self):
        return len(self._word_names)

    @property
    def corpus_size(self):
        if self._corpus_words is None:
            return sum(len(c) for c in self._chunk_tokens)
        else: 
            return len(self._corpus_words)

    @property
    def corpus_words(self):
        if self._corpus_words is None:
            self._corpus_words = tuple(t for chunk in self._chunk_tokens for t in chunk)
        return self._corpus_words

    @property
    def corpus_chunks(self):
        if self._corpus_chunks is None:
            self._corpus_chunks = tuple(ci for ci, chunk in enumerate(self._chunk_tokens) for t in chunk)
        return self._corpus_chunks

    @property
    def word_names(self):
        return iter(self._word_names)

    @property
    def chunk_names(self):
        return iter(self._chunk_names)

    def get_word_name(self, ix):
        return self._word_names[ix]

    def get_chunk_name(self, ix):
        return self._chunk_names[ix]

    def _load_chunk_data(self, path):
        vocabulary_set = set()
        for chunk_ix, (name, raw_chunk) in enumerate(self._chunk_iter(path)):
            token_list = self._tokenize(raw_chunk)

            new_words = [w for w in token_list if w not in vocabulary_set]
            nw_enum = enumerate(new_words, start=len(self._word_names))
            self._wordname_to_index.update((w, i) for i, w in nw_enum)
            self._word_names.extend(new_words)
            vocabulary_set.update(token_list)

            token_indices = [self._wordname_to_index[w] for w in token_list]
            self._chunk_tokens.append(token_indices)
            self._chunk_names.append(name)
            self._chunkname_to_index[name] = chunk_ix
        
    def _chunk_iter(self, path):
        for f in os.listdir(path):
            f = os.path.join(path, f)
            if os.path.isfile(f):
                with open(f, 'r') as chunkfile:
                    yield f, chunkfile.read()
    
    def _load_stopwords(self, path):
        if path:
            with open(path, 'r') as stopfile:
                return set(line.strip() for line in stopfile)
        else:
            return set()

    _punct_rex = re.compile(r'[^a-zA-Z\s]')
    def _tokenize(self, chunk, _punct_rex=_punct_rex):
        chunk = _punct_rex.sub('', chunk).lower()
        sw = self._stopwords
        return [word for word in chunk.split() if word not in sw]

class GibbsSampler(object):
    def __init__(self, corpus, n_topics, n_iterations, init_alphasum, init_betasum, progress_freq=50,
                 hyperparam_optimize=True, symmetric_alpha=False, symmetric_beta=False):
    
        # words (list of word labels per whole-corpus ix)
        # topics (list of topic labels per whole-corpus ix)
        # chunks (list of chunk labels per whole-corpus ix)

        self.progress_freq = progress_freq
        self.n_iterations = n_iterations
        self.n_topics = n_topics
        self.words = corpus.corpus_words
        self.chunks = corpus.corpus_chunks
        
        self.vocab_size = corpus.vocab_size
        self.corpus_size = corpus.corpus_size
        self.n_chunks = corpus.n_chunks
        
        self.topics = [random.randrange(0, self.n_topics) for i in xrange(self.corpus_size)]

        # define a hyperparameter alpha that, if symmetric, concentrates topics together
        #   within documents (high alpha) or spreads them apart between documents
        #   (low alpha). In other words, if we have 3 chunks with distributions 
        #   over 3 topics, those distributions will probably look like
        #   [.4, .3, .3], [.35, .3, .35], [.28, .37, .35]
        #   for high alpha, and will probably look more like
        #   [.7, .12, .18], [.1, .88, .02], [.12, .14, .74]
        #   for low alpha
        # a hyperparameter beta that does the same thing, except for topics
        #   over words
        # a companion to beta, betasum, which is the sum of beta[word] over all words

        self.hyperparam_optimize = hyperparam_optimize
        self.alphasum = init_alphasum
        self.alpha = [init_alphasum / self.n_topics] * self.n_topics  
        self.symmetric_alpha = symmetric_alpha
        self.betasum = init_betasum
        self.beta = [init_betasum / self.vocab_size] * self.vocab_size 
        self.symmetric_beta = symmetric_beta

        ctc, tcc, twc, ttc = self.count_tokens()
        self.chunk_topic_count = collections.Counter(ctc)
        self.topic_word_count = collections.Counter(twc)
        self.total_topic_count = collections.Counter(ttc)
        self.total_chunk_count = collections.Counter(tcc)

        self.multi_sample_ctc = collections.Counter()
        self.multi_sample_twc = collections.Counter()
        self.multi_sample_ttc = collections.Counter()
        self.multi_sample_tcc = collections.Counter()


    def count_tokens(self, topics=None):
        # number of words assigned to each topic-word pair
        # number of words assigned to each topic-chunk pair
        # number of words assigned to each topic overall

        if topics is None: 
            topics = self.topics
        chunk_topic_count = {}
        total_chunk_count = {c:0 for c in xrange(self.n_chunks)}
        for c, t in itertools.izip(self.chunks, topics):
            if (c, t) in chunk_topic_count:
                chunk_topic_count[c, t] += 1
            else:
                chunk_topic_count[c, t] = 1
            total_chunk_count[c] += 1

        topic_word_count = {}
        total_topic_count = {t:0 for t in xrange(self.n_topics)}
        for t, w in itertools.izip(topics, self.words):
            if (t, w) in topic_word_count:
                topic_word_count[t, w] += 1
            else:
                topic_word_count[t, w] = 1
            total_topic_count[t] += 1

        return chunk_topic_count, total_chunk_count, topic_word_count, total_topic_count

    def run(self):
        self.new_topic_distribution = [None] * self.n_topics
        for it in xrange(self.n_iterations):
            self.gibbs_sample()
            
            if not it % self.progress_freq: 
                print 'iteration {}'.format(it)
                print 'alpha sum:', self.alphasum, '\talpha avg:', self.alphasum / self.n_topics
                print 'beta sum: ', self.betasum, '\tbeta avg: ', self.betasum / self.vocab_size
            if not it % 10 and it > 200:
                self.multi_sample_tcc += self.total_chunk_count
                self.multi_sample_ctc += self.chunk_topic_count
                self.multi_sample_ttc += self.total_topic_count
                self.multi_sample_twc += self.topic_word_count

                if not it % 10 and self.hyperparam_optimize:   # TODO: Figure out correct way to estimate hyperparams -- old way on multi-sample works best right now
                   #self.old_estimate_hyperparameters()        #       although I did get some interesting results from the new method when the alpha values didn't 
                    self.estimate_hyperparameters()            #       all collapse to zero. In all my multi-param estimators, the beta value just keeps going up.
                                                               #       And I thought it was because I needed to average the samples, but that didn't seem to make a
                                                               #       difference...

                            # TODO: make sure that I've ironed out the stupid bugs involving vocab size vs. corpus size

    def estimate_hyperparameters(self):
        #alpha, alphasum = self.minka_fixed_point(self.multi_sample_ctc,
        #                                         self.multi_sample_tcc, 
        alpha, alphasum = self.minka_fixed_point(self.chunk_topic_count,
                                                 self.total_chunk_count, 
                                                 self.alpha, 
                                                 self.alphasum)
        self.alpha = alpha
        self.alphasum = alphasum

        #beta, betasum  = self.minka_fixed_point(self.multi_sample_twc, 
        #                                        self.multi_sample_ttc,
        beta, betasum  = self.minka_fixed_point_symmetric(self.topic_word_count,
                                                self.total_topic_count,
                                                self.beta,
                                                self.betasum)
        self.beta = beta
        self.betasum = betasum

    _eps=sys.float_info.epsilon * 10 ** 5
    @staticmethod
    def minka_fixed_point(pair_counts, total_counts, params, paramsum, min_val=_eps):
        K = len(params)
        M = len(total_counts)
        params = params[:]
        all_total_counts = [total_counts[m] for m in xrange(M)]
        for k, old_param_k in enumerate(params):
            param_k = old_param_k
            ratio = 0
            count = 0
            all_pair_counts = [pair_counts[m, k] for m in xrange(M)]
            while (ratio < 0.95 or ratio > 1.05) and param_k > min_val:
                
                num = sum([0 if pc <= 0 else float(pc) / (pc - 1 + param_k) for pc in all_pair_counts])
                den = sum([0 if tc <= 0 else float(tc) / (tc - 1 + paramsum) for tc in all_total_counts])
                ratio = num / den
                
                paramsum -= param_k
                param_k *= ratio
                paramsum += param_k
                
                break

                count += 1

            params[k] = param_k

        return params, paramsum

    @staticmethod
    def minka_fixed_point_symmetric(pair_counts, total_counts, params, paramsum, min_val=_eps):
        K = len(params)
        M = len(total_counts)
        param = paramsum / K
        all_pair_counts = [pair_counts[m, k] for m in xrange(M) for k in xrange(K)]
        all_total_counts = [total_counts[m] for m in xrange(M)]
        ratio = 0
        count = 0
        while (ratio < 0.95 or ratio > 1.05) and paramsum > min_val:
            num = sum([float(pc) / (pc - 1 + param) if pc > 0 else 0 for pc in all_pair_counts])
            den = K * sum([float(tc) / (tc - 1 + paramsum) if tc > 0 else 0 for tc in all_total_counts])
            ratio = num / den
            
            param *= ratio
            paramsum = param * K

            break

            count += 1
            if count > 500:
                print count
                print paramsum
                break

        #print "symmetric parameter fixed-point reached after {} iterations".format(count)
        return [param] * K, paramsum

    def old_estimate_hyperparameters(self):
        
        tpmeans, tpvars = self.dirichlet_mean_var(self.multi_sample_ctc,
        #tpmeans, tpvars = self.dirichlet_mean_var(self.chunk_topic_count,
                                                  self.n_chunks, self.n_topics)
        alpha, alphasum = self.dirichlet_params(tpmeans, tpvars)
        self.alphasum = alphasum
        if self.symmetric_alpha:
            self.alpha = [self.alphasum / self.n_topics] * self.n_topics
        else:
            self.alpha = alpha

        wordmeans, wordvars = self.dirichlet_mean_var(self.multi_sample_twc,
        #wordmeans, wordvars = self.dirichlet_mean_var(self.topic_word_count, 
                                                      self.n_topics, self.vocab_size)
        beta, betasum = self.dirichlet_params(wordmeans, wordvars)
        self.betasum = betasum
        if self.symmetric_beta:
            self.beta = [self.betasum / self.vocab_size] * self.vocab_size
        else:
            self.beta = beta

    def dirichlet_mean_var(self, count_dict, max_i, max_j):
        '''Traverse a sparse matrix, interpreting each row as a sampling of
        a categorical distribution of tokens with probabilities themselves 
        drawn from a Dirichlet distribution. These stats (means and variances
        over all columns for each row) will help us infer a reasonable
        hyperparameter vector using the initial estimation method described in 
        Ronning 2009. I think. (At some point, this will be replaced by a 
        proper estimation algorithm.)'''

        rowcounts = [0.0] * max_i
        for (i, j), count in count_dict.iteritems():
            rowcounts[i] += count

        means = [0.0] * max_j
        for (i, j), count in count_dict.iteritems():
            means[j] += float(count) / rowcounts[i]
        
        for j, m in enumerate(means):
            means[j] = m / max_i

        variances = [0.0] * max_j
        for (i, j), count in count_dict.iteritems():
            variances[j] += ((float(count) / rowcounts[i]) - means[j]) ** 2

        for i, v in enumerate(variances):
            variances[i] = float(v) / (max_i - 1)   # iterate or use list comprehension? TODO: test this

        return means, variances

    def dirichlet_params(self, means, variances):
        '''Estimate the parameters of a dirichlet distribution based
        on mean and variance. This is derived directly from the sufficient
        statistics listed on the wikipedia page. However, it produces 
        estimates that are off by about an order of magnitude compared to
        MALLET's estimates. That's an order of magnitude too high for 
        the alpha values, and an order of magnitude too low for the 
        beta values. My hypothesis is that it's suffering mean
        bias because of the sparsity of the counts. (That is, all those
        zero counts would increase to some nonzero value after enough
        samples, but there aren't enough samples, so the mean is
        artificially skewed towards the most common values.) But I 
        actually have no clue. If I ran this on a larger corpus would
        the results be more in line with MALLET's? I don't know.
        I need to write a real estimator, but that means I have 
        to learn things. Ugh.'''

        # I think I should check this against that estimation initialization method in the article from 89
        # I'm pretty sure it does the same thing but not certain.

        antimeans = [1 - m for m in means]
        Qvals = [v / u for v, u in itertools.izip(variances, antimeans)]

        param_sum = 1.0 / sum(Qvals) - 1
        
        # I am extremely confused. Where did I get this?
        #params = [(param_sum * param_sum + param_sum) * q for q in Qvals]

        # And why can't I just do this?                     # TODO: My understanding of the dirichlet distribution values suggests that
        params = [m * param_sum for m in means]             #       we ought to just be able to multiply the dirichlet params by the 
                                                            #       sum. But the formula I derived does something totally different
        return params, param_sum                            #       that produces results with a different shape. Correlated? Not sure.
    
                                                            # TODO: Now I have to try to implement the other Minka fixed-point trick
                                                            #       and see if IT works and produces values that make sense.
    def gibbs_sample_sparse(self):
        '''One iteration of LDA Gibbs Sampling as described by Darling (2011)
        with a few tricks to maintain sparse "arrays." This is equivalent to 
        MALLET's SimpleLDA class as well. ''' 
        
        # TODO: write a gs function based on SparseLDA (cf. yao mimno mcallum 2009)

        for i in xrange(self.corpus_size):
            # get chunk/topic/word values for current index in corpus
            # then decrement the corresponding token counts

            c = self.chunks[i]
            t = self.topics[i]
            w = self.words[i]
            self.chunk_topic_count[c, t] -= 1
            self.topic_word_count[t, w] -= 1
            self.total_topic_count[t] -= 1

            # maintain sparsity -- this gives a ~30% speedup with pypy

            if self.chunk_topic_count[c, t] == 0: 
                del self.chunk_topic_count[c, t]
            if self.topic_word_count[t, w] == 0: 
                del self.topic_word_count[t, w]

            # calculate p(t(i) | C, W) (with i-th counts from W, C omitted) and sample

            for tp in xrange(self.n_topics):           # note: the alpha bottom term is constant (sum of chunk sizes doesn't change!) so we omit here
                ct_top = self.alpha[tp] + (0 if (chunk, tp) not in self.chunk_topic_count
                                             else self.chunk_topic_count[chunk, tp])
                tw_top = self.beta[word] + (0 if (tp, word) not in self.topic_word_count
                                           else self.topic_word_count[tp, word])
                tw_bottom = self.total_topic_count[tp] + self.betasum
                self.new_topic_distribution[tp] = ct_top * tw_top * 1.0 / tw_bottom

            t = self.categorical_sample()
            
            # save new topic value and increment corresponding counts

            self.topics[i] = t
            if (c, t) in self.chunk_topic_count:
                self.chunk_topic_count[c, t] += 1
            else:
                self.chunk_topic_count[c, t] = 1
            if (t, w) in self.topic_word_count:
                self.topic_word_count[t, w] += 1
            else:
                self.topic_word_count[t, w] = 1
            self.total_topic_count[t] += 1

    def gibbs_sample(self):
        '''One iteration of LDA Gibbs Sampling as described by Darling (2011)
        with a few tricks to maintain sparse "arrays." This is equivalent to 
        MALLET's SimpleLDA class as well. ''' 

        for i in xrange(self.corpus_size):
            # get chunk/topic/word values for current index in corpus
            # then decrement the corresponding token counts

            c = self.chunks[i]
            t = self.topics[i]
            w = self.words[i]
            self.chunk_topic_count[c, t] -= 1
            self.topic_word_count[t, w] -= 1
            self.total_topic_count[t] -= 1

            # maintain sparsity -- this gives a ~30% speedup with pypy

            if self.chunk_topic_count[c, t] == 0: 
                del self.chunk_topic_count[c, t]
            if self.topic_word_count[t, w] == 0: 
                del self.topic_word_count[t, w]

            # calculate p(t(i) | C, W) (with i-th counts from W, C omitted) and sample
            
            for tp in xrange(self.n_topics):           # note: the alpha bottom term is constant (sum of chunk sizes doesn't change!) so we omit here
                ct_top = self.alpha[tp] + (0 if (c, tp) not in self.chunk_topic_count
                                             else self.chunk_topic_count[c, tp])
                tw_top = self.beta[w] + (0 if (tp, w) not in self.topic_word_count
                                           else self.topic_word_count[tp, w])
                tw_bottom = self.total_topic_count[tp] + self.betasum
                self.new_topic_distribution[tp] = ct_top * tw_top * 1.0 / tw_bottom

            t = self.categorical_sample()
            
            # save new topic value and increment corresponding counts

            self.topics[i] = t
            if (c, t) in self.chunk_topic_count:
                self.chunk_topic_count[c, t] += 1
            else:
                self.chunk_topic_count[c, t] = 1
            if (t, w) in self.topic_word_count:
                self.topic_word_count[t, w] += 1
            else:
                self.topic_word_count[t, w] = 1
            self.total_topic_count[t] += 1

    def categorical_sample(self):
        probs = self.new_topic_distribution
        total = sum(probs)
        val = random.random() * total
        cum = 0
        for i, p in enumerate(probs):
            cum += p
            if val < cum:
                return i

if __name__ == '__main__':
    
    lda_parser = argparse.ArgumentParser('lda.py', description='An implementation of Latent Dirichlet Allocation in pure Python.')
    lda_parser.add_argument('-I', '--input-dir', metavar='input_directory', required=True, help='Path to a folder full of text files to be analyzed, treating each text file as a document.')
    lda_parser.add_argument('-S', '--stopwords', metavar='stopword_file', required=True, help='Path to text file full of stop words, one word per line.')
    lda_parser.add_argument('-n', '--num-topics', metavar='number_of_topics', type=int, default=50, help='Number of topics. Defaults to 50')
    lda_parser.add_argument('-i', '--num-iterations', metavar='number_of_iterations', type=int, default=200, help='Number of iterations. Defaults to 200')
    lda_parser.add_argument('-a', '--init-alphasum', metavar='initial_alpha', type=float, default=50.0, help='Initial value for the sum of all alpha values, symmetrically divided among topics. For asymmetric optimization, select the --optimize option.')
    lda_parser.add_argument('-b', '--init-betasum', metavar='initial_beta', type=float, default=100.0, help='Initial value for the sum of all beta values, symmetrically divided among words. [Note: Using the MALLET convention of specifying this as a per-word value might make more sense after all, since the exact number of words in a corpus is unknown, or at least annyoing to determine, which means that the effective value per word varies by corpus.]')
    lda_parser.add_argument('-f', '--report-frequency', metavar='number_of_iterations', type=int, default=50, help='Number of iterations to wait before printing a progress update. Defaults to 50.')
    lda_parser.add_argument('-k', '--save-keyfile', metavar='key_file', default='.temp.lda.keys', help='Filename under which to save topic key terms. Defaults to a hidden temporary file `.temp.lda.keys`.')
    lda_parser.add_argument('-c', '--save-compfile', metavar='composition_file', default='.temp.lda.composition', help='Filename under which to save document composition data. Defaults to a hidden temporary file `.temp.lda.composition`.')
    lda_parser.add_argument('-o', '--optimize', action='store_true', default=False, help='Optimize hyperparameters. Alpha is asymmetric by default, but beta always remains symmetric. For symmetric alpha estimation, select the --symmetric-alpha. option.')
    lda_parser.add_argument('-A', '--symmetric-alpha', action='store_true', default=False, help='Keep alpha symmetric during optimization. This implies the --optimize option.')

    args = lda_parser.parse_args()

    # load the corpus and set up the sampler

    corpus = Corpus(args.input_dir, args.stopwords)   
    gibbs = GibbsSampler(corpus,
                         args.num_topics,
                         args.num_iterations, 
                         args.init_alphasum, 
                         args.init_betasum,
                         args.report_frequency,
                         hyperparam_optimize=args.optimize,
                         symmetric_alpha=args.symmetric_alpha,
                         symmetric_beta=True)    # This should always be true. See below. 

    gibbs.run()

    # Took me a long time to understand all the details here. Beta is _symmetric_ in MALLET -- always! The Wallach article on hyperpriors indicates that 
    # the best methods are asymmetric alpha, asymmetric beta AND asymmetric alpha, symmetrica beta. And the AS option is actully slightly better (it seems)
    # than the AA option. For that reason, The "use-symmetric-alpha" option that MALLET provides is the _only_ way to manipulate symmetry in MALLET -- 
    # your only options are SS and AS. Asymmetric beta isn't even an option. So that explains why the beta value you pass is a per-token value, while the
    # alpha value you pass is a sum. I don't know why I was so convinced that MALLET uses asymmetric priors for both values.

    # So the alpha you pass in is divided amongst the topics equally (or unequally), while the beta you pass in is _summed_ across every token for 
    # those moments when the sum of beta is required. This explains why the `s` term in the sparse-optimized update equation is constant. Well, sort of.
    # I still don't get why n(t) supposedly doesn't change when you subtract the current topic assignment. If the bottom term is constant, why not drop it?

    # My current answer to that last question is an intellectual punt: the paper was missing a small detail. I can't make sense of it otherwise. The 
    # denominator for s, r, and q has to be updated for every token during gibbs sampling. It's just that those updates can be done in constant time using 
    # some complicated schenanegans. (For a given topic assignment to a particular token, you have to subtract the numerator for that topic from the
    # numerator sum, then divide that sum by the denominator for that topic, then multiply it by one less than the denominator for that topic, and then 
    # add the numerator for that topic back in. The result is the sum of all numerators for all tokens minus that one token. Then you have to divide the
    # denominator for the whole set of summed numerators by the numerator for that topic, and then multiply it by one less than that number. Then when
    # you want to reassign the topic, you do the same in reverse.) This is similar to the update process for r and q, though they all happen at different
    # times. Apaprently Q has to be recalculated more frequently. It's all too much too work to do right now. Maybe over the summer when I'm not applying
    # for jobs and teaching a 3/3 load in _addition_ to writing.

    
    #These are fugly.

    gibbs.chunk_topic_count = gibbs.multi_sample_ctc
    gibbs.topic_word_count = gibbs.multi_sample_twc

    with open(args.save_compfile, 'w') as comp:
        for c, cname in enumerate(corpus.chunk_names):
            topic_counts = [(gibbs.chunk_topic_count.get((c, t), 0), t) 
                            for t in xrange(gibbs.n_topics)]
            topic_counts.sort(reverse=True)
            total_topic_count = sum(c[0] for c in topic_counts)
            
            fprint_c('{:30}'.format(cname), comp)
            for count, label in topic_counts:
                fprint_c('{:4}'.format(label), comp)
                fprint_c('{:<12.12f}'.format(count * 1.0 / total_topic_count), comp)
            fprint('', comp)

    with open(args.save_keyfile, 'w') as keys:
        alphas_topics = sorted(((a, t) for t, a in enumerate(gibbs.alpha)), reverse=True)
        #alphas_topics = ((a, t) for t, a in enumerate(gibbs.alpha))
        for alpha, topic in alphas_topics:
            word_counts = [(gibbs.topic_word_count.get((topic, w), 0), w) 
                           for w in xrange(gibbs.vocab_size)]
            word_counts.sort(reverse=True)
            
            fprint_c('{}\t'.format(topic), keys)
            fprint_c('{:10.5}\t'.format(alpha), keys)
            for i in xrange(200):
                count, word = word_counts[i]
                fprint_c(corpus.get_word_name(word), keys)
            fprint('', keys)



