from __future__ import division
import os
import numpy as np
import operator
import math
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
from sklearn.cluster import AgglomerativeClustering
import argparse
from multiprocessing import Pool
import random
from scipy.optimize import curve_fit
import pandas as pd
import itertools
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA



"""
------------------------------------------------------------------------------------
MIT License

Copyright (c) 2019 Iskander Said

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------------

A modular command line parser for the coder on the go. 

Simply add or remove arguments depending on your program. The code implementation is quite straightforward simply
write:

def main():
#Call command line class
    myCommandLine = CommandLine()
    #Call a commandline argument
    myArg = myCommandLine.args.myArg

main()

Done. Boom. 
"""


class CommandLine():
    def __init__(self, inOpts=None):
        self.parser = argparse.ArgumentParser(
            description='An analysis tool for SCI-SEQ WGS data.',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            # Allows the epilog to be formatted in the way I want!!
            epilog=('''       
                                           
            Currently is able to only handle Drosophila Melanogaster data. This tool is designed to detection recombination breakpoints
            in single cell sequencing data using an HMM. Then it will cluster these cells by their recombination breakpoints to merge similar individuals.

              '''),
            add_help=True,  # default is True
            prefix_chars='-',
            usage='%(prog)s [options] -option1[default] <input >output'
        )  ###Add or remove optional arguments here
        self.parser.add_argument("-p", "--polarize", action="store_true",
                                 help="Call this option to polarize the SNP array from the single cell simulated SNP array to a P1, P2, error polarization.",
                                 default=False)
        self.parser.add_argument("-s", "--snp", type=str, action="store", nargs="?",
                                 help="The SNP array to be input for analysis.",
                                 default='/home/iskander/Documents/Barbash_lab/mDrive/SIM_DATA/SPARSE_simData.npy')

        self.parser.add_argument("-t", "--threads", type=int, action="store", nargs="?",
                                 help="Number of threads to be specified if multi-threading is to be used.",
                                 default=1)
        self.parser.add_argument("-rm", "--rmap", type=str, action='store', nargs='?', help='Recombination map', default='/home/iskander/Documents/Barbash_lab/mDrive/dmel_2xRate.rmap.bed')
        self.parser.add_argument("-i", "--interval", type=float, action='store', nargs='?', default=.99)
        self.parser.add_argument('-d', '--distance', type=float, action='store', nargs='?', default=.2)
        self.parser.add_argument('-r', '--rate', type=float, action='store', nargs='?', default=2.46)

        if inOpts is None:  ## DONT REMOVE THIS. I DONT KNOW WHAT IT DOES BUT IT BREAKS IT
            self.args = self.parser.parse_args()
        else:
            self.args = self.parser.parse_args(inOpts)




class analyzeSEQ:

    def __init__(self):
        self.polarized_samples = [] #The array that contains all of our sample data VERY IMPORTANT

    ##### Set parameters for the multinomials ###

        #P1 homozygous
        self.pi_p1_1 = 0.9993 #P1 allele in P1 homozygoous region
        self.pi_p1_2 = 0.0002 #P2 allele in P1 homozygous region
        self.pi_p1_3 = 0.0005 #Seq error in P1 homozygous region
        #P1/P2 heterozygous
        self.pi_p2_1 = 0.5005 #P1 allele in P1/P2
        self.pi_p2_2 = 0.499 #P2 allele in P1/P2
        self.pi_p2_3 = 0.0005 #Seq error in P1/P2



        ###Probability states for the HMM###

        self.init_chrom_states = {'p1':np.log(0.5), 'p2':np.log(0.5)}
        self.state_probs = {'p1':[np.log(self.pi_p1_1), np.log(self.pi_p1_2), np.log(self.pi_p1_3)],
                       'p2': [np.log(self.pi_p2_1), np.log(self.pi_p2_2), np.log(self.pi_p2_3)]
                       }
        """
        Probability of transition is the probability of an odd number of recombination breakpoints given the genetic distance between two SNPs.
        To get around log probabilities not handling 0 I am going to replace 0s with lowest possibly float in Python so we effectively compute 0 prob but w/o raising errors
        """

        self.transitions_probs ={'p1':{'p1': lambda x: np.log( max( (1 - ((1 - math.exp(-((2 * x) / 100))) / 2)), sys.float_info.min) ),
                                       'p2': lambda x: np.log( max( ((1 - math.exp(-((2 * x) / 100))) / 2), sys.float_info.min) ) },

                                 'p2':{'p1': lambda x: np.log( max( ((1 - math.exp(-((2 * x) / 100))) / 2), sys.float_info.min) ),
                                       'p2': lambda x: np.log( max( (1 - ((1 - math.exp(-((2 * x) / 100))) / 2)), sys.float_info.min) ) }
                                 }

        ### Important data structures for SD detection and parameter estimation
        self.heterochromatin = {
            # A map of the freely recombining intervals of euchromatin within each drosophila chromosome arm in Mb

        }
        self.cM_boundaries = {

        }
        self.paintedGenome = []
        self.fitted_allele_frequencies = []
        self.pseudoBulk_data = []
        self.bulk_params = []
        self.bulk_errors = []
        self.param_estimates = []
        self.param_errs = []
        self.recomb_rate = 2.46
        self.rmap = None
        self.chroms = []

    def get_cM(self, pos):
        """
        Calculate the transition probabilty as a function of recombination distance between each SNP marker in array of emissions as calculated by
        https://petrov.stanford.edu/cgi-bin/recombination-rates_updateR5.pl#Method

        Since I have a drosophila recombination map I can convert my BP positions to cM and then get the distance in cM between the two positions.
        This distance is then fed into a lambda function instantiate in the __init__ method which is called within my HMM methods.


        :return:
        """

        #We use this function to compute the distance in cM from the two SNPs

        #Just use genome wide average, although accounting for heterochromatin would be ideal

        RATE = self.recomb_rate

        #This method will compute the genetic distance as simply the distance given the genome wide average recombination rate
        SNP_1 = pos[0]
        SNP_2 = pos[1]

        genetic_dist = abs(SNP_1 - SNP_2) * RATE

        """
        These sets of methods attempts to calculate the genetic distance using the melanogaster recombination map function and accounting for centromeric heterochromatin
        
        cM_pos = []
        for snp in range(2):
            #Filter for SNPs that lie in the heterochromatin if they do use the cM distance that is the border. This way any two SNPs within heterochromatin will have distance of 0
            if pos[snp] <= self.heterochromatin[arms[snp]][0]:
                cM = self.cM_map(arms[snp])(self.heterochromatin[arms[snp]][0])

            elif pos[snp] >= self.heterochromatin[arms[snp]][1]:
                cM = self.cM_map(arms[snp])(self.heterochromatin[arms[snp]][1])
            else:
                cM = self.cM_map(arms[snp])(pos[snp])
            cM_pos.append(cM)

        if arms[0] != arms[1]: #If the SNPs cross the pericentromere boundary
            if pos[0] >= self.heterochromatin[arms[0]][1] and pos[1] <= self.heterochromatin[arms[1]][0]: #When both SNPs are in the pericentric heterochromatin we call the recombination rate as 0
                genetic_dist = 0

            elif pos[0] <= self.heterochromatin[arms[0]][1] and pos[1] <= self.heterochromatin[arms[1]][0]: #When the right arm is in the pericentric heterochromatin but the left arm SNP isn't then we use
                #the boundary from the left arms heterochromatin to not artificially inflate the recombination distance
                cM_pos[1] = self.cM_map(arms[0])(self.heterochromatin[arms[0]][0])
                genetic_dist = abs(cM_pos[0] - cM_pos[1])

            elif pos[0] >= self.heterochromatin[arms[0]][1] and pos[1] >= self.heterochromatin[arms[1]][0]: #As the above elif but with the arms reversed
                cM_pos[0] = self.cM_map(arms[1])(self.heterochromatin[arms[1]][0])
                genetic_dist = abs(cM_pos[0] - cM_pos[1])

            elif pos[0] < self.heterochromatin[arms[0]][1] and pos[1] > self.heterochromatin[arms[1]][0]:
                # If two SNPs are on either side of the heterochromatin, but not within it the genetic distance is much smaller than what we would initially estimate as the probability of recombination
                # in that interval must not include the heterochromatin
                het_distance = abs(self.cM_map(arms[0])(self.heterochromatin[arms[0]][1]) - self.cM_map(arms[1])(self.heterochromatin[arms[1]][0])) #This is the cM distance in the heterochromatin
                genetic_dist = max(abs(cM_pos[0] - cM_pos[1]) - het_distance, 0)#This subtracts the cM distance in heterochromatin so as to not artificially inflate the genetic distance

        else:
            #Get distance in cM and then return it
            genetic_dist = abs(cM_pos[0] - cM_pos[1])
        """

        return genetic_dist

    def decode(self, pointers, snp_matrix):
        """
        Use our maximum likelihood algorithm to find the point where our state space changes from heterozygous to homozygous so we can infer the SNPs of each of our individuals

        :param pointers:
        :param snp_matrix:
        :return:
        """

        rbp_labels = [[] for p in range(len(snp_matrix))]
        rbps = [[] for p in range(len(snp_matrix))]


        all_chr_arms = [snp_matrix[arm][:,0]/1000000 for arm in range(len(snp_matrix))]
        chr_index = 0

        for traceback in pointers:
            #iterate through pointers for each chromosome

            for position in range(1, len(traceback)):
                #Check for a change in state from P1/P1 to P1/P2
                if traceback[position-1] != traceback[position]:
                    breakpoint = all_chr_arms[chr_index][position-1] + abs(all_chr_arms[chr_index][position] - all_chr_arms[chr_index][position-1])/2 #take midpoint from the switch over position
                    rbps[chr_index].append(breakpoint)
                    rbp_labels[chr_index].append((traceback[position-1], traceback[position]))
            if len(rbps[chr_index]) == 0:
                rbp_labels[chr_index].append(traceback[position])
            chr_index += 1


        return rbps, rbp_labels

    def hmmFwBw(self, snp_input):


        states = ['p1', 'p2']
        chr_posteriors = []


        all_chr_arms = [snp_input[arm][:, 0] / 1000000 for arm in range(len(snp_input))]


        # Fwd Bckwd

        for chrom in range(len(snp_input)):  # consider each chromosome as a single sequence


            chrom_length = len(snp_input[chrom])
            chromosome_arm = snp_input[chrom]

            fMatrix = np.zeros(shape=(2, chrom_length))

            # First state
            # P(state) * P(emission|state)
            init_emission = int(snp_input[chrom][0][1])

            for state in range(2):
                fMatrix[state][0] = self.init_chrom_states[states[state]] + self.state_probs[states[state]][
                    init_emission]


            # Proceed along sequence of SNPs and compute probabilities

            for SNP in range(1, len(chromosome_arm)):

                pos_snps = [all_chr_arms[chrom][SNP], all_chr_arms[chrom][SNP - 1]]
                dist = self.get_cM(pos=pos_snps)

                emission = int(chromosome_arm[SNP][1])
                for state in range(2):

                    # P(emission|state) * Sum(P(transition to state) * P(state -1), P(transition to state) * P( state -1))
                    sumScore = self.sumLogProb(self.transitions_probs['p1'][states[state]](dist) + fMatrix[0][SNP-1],
                                               self.transitions_probs['p2'][states[state]](dist) + fMatrix[1][SNP-1])
                    newScore = self.state_probs[states[state]][emission] + sumScore
                    fMatrix[state][SNP] = newScore


            #Bckwd algorithm
            bMatrix = np.zeros(shape=(2, chrom_length))
            # initial state
            for state in range(2):
                bMatrix[state][-1] = np.log(1)
            # backward prob
            # Almost same as forward except backward
            for SNP in range(chrom_length-1, 0, -1):  # loop backwards starting at -1 of seq
                SNP = SNP - 1  # correct for index

                pos_snps = [all_chr_arms[chrom][SNP], all_chr_arms[chrom][SNP + 1]]

                dist = self.get_cM(pos=pos_snps)


                # First loop is to go through sequence
                for state in range(2):
                    # Sum(P(transition to state) * P(state + 1) * P(state +1),
                    #  P(transition to state) * P( state + 1) * P(emission + 1|state+1) )
                    emission = int(chromosome_arm[SNP+1][1])

                    bestScore = self.sumLogProb(
                        self.state_probs['p1'][emission] + self.transitions_probs['p1'][states[state]](dist) + bMatrix[0][SNP + 1],

                        self.state_probs['p2'][emission] + self.transitions_probs['p2'][states[state]](dist) + bMatrix[1][SNP + 1])

                    bMatrix[state][SNP] = bestScore


            # Posterior probability
            numerator = bMatrix + fMatrix  # f(i) * b(i)
            # To calculate denominataor
            denominator = []

            for column in range(len(bMatrix[0, :])):
                denominator.append(self.sumLogProb(numerator[0][column], numerator[1][column]))



            # posteriorMatrix = numerator-denomMatrix
            P1 = numerator[0] - denominator
            P2 = numerator[1] - denominator
            posteriorMatrix = np.asarray([P1, P2])
            chr_posteriors.append(posteriorMatrix)


        #sys.stdout.write('Posterior probability matrix constructed.\n')
        return chr_posteriors

    def draw(self, posterior_matrix, snp_matrix, predicted_rbps, truth, markers, title='TestID'):
        chr_decoding = {0:'Chr2', 1:'Chr3', 2:'ChrX'}

        chr_2 = np.concatenate((snp_matrix[0][:, 0], snp_matrix[1][:, 0] + 23000000)) / 1000000
        chr_3 = np.concatenate((snp_matrix[2][:, 0], snp_matrix[3][:, 0] + 24500000)) / 1000000
        chr_x = snp_matrix[4][:, 0] / 1000000
        all_chr_arms = [chr_2, chr_3, chr_x]


        chr2_alleles = np.concatenate((snp_matrix[0], snp_matrix[1]))[:,1]
        chr3_alleles = np.concatenate((snp_matrix[2], snp_matrix[3]))[:,1]
        chrX_alleles = snp_matrix[4][:,1]

        all_alleles = [chr2_alleles, chr3_alleles, chrX_alleles]
        with sns.axes_style('whitegrid'):
            fig = plt.figure(figsize=(20, 10))
            axs = fig.subplots(3,2)


        for arm in range(3):
            #left_interval = all_chr_arms[arm][int(interval[arm][0])]
            #right_interval = all_chr_arms[arm][int(interval[arm][1])]

            sns.scatterplot(all_chr_arms[arm], all_alleles[arm], ax=axs[arm][1])
            sns.lineplot(all_chr_arms[arm], posterior_matrix[arm][0] , ax=axs[arm][0]) #Draw posterior prob of P1/P1

            #sns.lineplot(all_chr_arms[arm], posterior_matrix[arm][1], ax=axs[arm][0])
            axs[arm][0].set_title('{0}'.format(chr_decoding[arm]))
            axs[arm][0].set_ylabel('log(odds)')
            axs[arm][0].set_xlabel('Position (Mb)')



            axs[arm][1].set_yticks([0,1,2])
            axs[arm][1].set_yticklabels(['P1', 'P2', 'Error'])
            other_legend = []
            legend = ['P1/P1']

            for pred in predicted_rbps[arm,:]:
                if pred != 0:
                    legend = legend + ['Predicted Breakpoint: {0:.0f} bp'.format(pred * 1000000)]
                    axs[arm][0].axvline(pred, linestyle='--', color='red')

            for bp in truth[arm]:
                other_legend = other_legend + ['True Breakpoint: {0:.0f} bp'.format(bp * 1000000)]
                legend = legend + ['True Breakpoint: {0:.0f} bp'.format(bp * 1000000)]
                axs[arm][0].axvline(bp, linestyle='--', color='blue')
                axs[arm][1].axvline(bp, linestyle='--', color='blue')
            other_legend = other_legend + ['Parental alleles']
            #axs[arm][0].axvline(left_interval, linestyle='--', color='green')
            #axs[arm][0].axvline(right_interval, linestyle='--', color='green')


            axs[arm][0].legend(legend)
            axs[arm][1].legend(other_legend)


        plt.suptitle('Recombination Breakpoint Predictions ({0} SNPs)'.format(markers))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_name = os.path.join('/home/iskander/Documents/MEIOTIC_DRIVE/', '{0}.png'.format(title))
        plt.savefig(plot_name, dpi=200)
        plt.close()

    def hmmViterbi(self, snp_input):

        """
        This function will perform our viterbi decoding for our recombination breakpoint detection. Depending on how fuzzy the breakpoint positions end up looking
        it may just be better to use the posterior distribution of the states and take like an area around the inflection point of the changes in posterior probabilities.


        :param snp_input:
        :return:
        """

        #sys.stdout.write('Computing Viterbi decoding through HMM state space...')


        arm_pointers = []
        states = ['p1', 'p2']
        chr_posteriors = []

        all_chr_arms = [snp_input[arm][:, 0] / 1000000 for arm in range(len(snp_input))]

        # Viterbi

        for chrom in range(len(snp_input)):  # consider each chromosome as a single sequence

            chrom_length = len(snp_input[chrom])
            chromosome_arm = snp_input[chrom]
            hmmMatrix = np.zeros(shape=(2, chrom_length))
            tracebackMatrix = np.zeros(shape=(2, chrom_length))

            #First state
            # P(state) * P(emission|state)

            init_emission = int(snp_input[chrom][0][1])

            for state in range(2):
                hmmMatrix[state][0] = self.init_chrom_states[states[state]] + self.state_probs[states[state]][init_emission]
                tracebackMatrix[state][0] = state


            #Proceed along sequence of SNPs and compute probabilities and fill traceback matrix

            for SNP in range(1, len(chromosome_arm)):

                pos_snps = [all_chr_arms[chrom][SNP], all_chr_arms[chrom][SNP - 1]]

                dist = self.get_cM(pos=pos_snps)

                emission = int(chromosome_arm[SNP][1])
                for state in range(2):

                    # P(emission|state) * Max (P(transition to state) * P(state -1), P(transition to state) * P( state -1)
                    bestScore = max((self.transitions_probs['p1'][states[state]](dist) + hmmMatrix[0][SNP - 1], 0),
                                    (self.transitions_probs['p2'][states[state]](dist) + hmmMatrix[1][SNP - 1], 1),
                                    key=operator.itemgetter(0))


                    newScore = (self.state_probs[states[state]][emission] + bestScore[0], bestScore[1])
                    hmmMatrix[state][SNP] = newScore[0]
                    tracebackMatrix[state][SNP] = newScore[1]

            # Traceback matrix and probability matrix have been instantiated now we will use traceback matrix to point back and call hidden states
            startPos = int(max(zip(hmmMatrix[:, -1], tracebackMatrix[:, -1]))[1])

            pointers = [startPos]

            for hiddenState in range(len(chromosome_arm) - 2, -1, -1):
                # hidden state 0
                if tracebackMatrix[startPos][hiddenState] == 0:
                    startPos = 0
                    pointers.append(0)
                # hidden state 1
                else:
                    startPos = 1
                    pointers.append(1)
            pointers.reverse()

            arm_pointers.append(pointers)

        return arm_pointers

    def sumLogProb(self, a, b):
        if a > b:

            return a + np.log1p(math.exp(b - a))

        else:

            return b + np.log1p(math.exp(a - b))

    def load_SNP_array(self, snp_array, path, encoding='UTF-8'):

        snp_File = os.path.join(path, snp_array)
        SNPs = np.load(snp_File, encoding=encoding, allow_pickle=True)
        sys.stdout.write('SNP input loaded... {0} cells read in...\n'.format(len(SNPs)))
        return SNPs

    def cluster_RBPs(self, predictions, distance, threads, iter_=5):
        """

        This method will use a form of clustering algorithm along with whatever dimensionality reduction methods I deem necessary to cluster my individuals properly.


        :return:
        """
        """
        sys.stdout.write("Optimizing tSNE embedding under {0} iterations\n".format(iter_))

        #tSNE manuals recommend doing PCA on the data if the number of dimensions is extremely high:
        pca_components = PCA(n_components=50).fit_transform(predictions)

        #instantiate tsne_model:
        alpha = len(self.polarized_samples) / 3
        tsne_model = TSNE(n_components=2, perplexity=50, init='pca', early_exaggeration=alpha)

        #In order to run the process multithreaded we have to create a list of the features data structure that will be reduced by tSNE iteratively
        data_array = []
        for i in range(iter_):
            data_array.append(pca_components)

        #We perform  dimensionality reduction process to learn the underlying manifold of the data and then represent this in 3 dimensions

        myPool = Pool(processes=threads)
        model_embeddings = myPool.map(tsne_model.fit, data_array)


        # Now find the optimal kl_divergence and use that embedding
        optima = np.inf
        opt_embed = False
        for model in range(iter_):
            if model_embeddings[model].kl_divergence_ < optima:
                optima = model_embeddings[model].kl_divergence_
                opt_embed = model_embeddings[model].embedding_

        sys.stdout.write('tSNE model determined for data set with minimum KL divergence of {0:.4f}\n'.format(optima))

        sys.stdout.write('DBSCAN clustering with epsilon = {0}...\n'.format(distance))
        #Next we cluster our data using DBSCAN a density based approach
        """

        cluster_pred = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=distance, affinity='euclidean', linkage='ward').fit_predict(predictions)
        #cluster_pred = DBSCAN(eps=distance, min_samples=1, n_jobs=threads).fit(opt_embed).labels_

        #### Get all cluster predictions and cells into a table:
        all_clusters = {}
        for cell in range(len(cluster_pred)):
            if cluster_pred[cell] not in all_clusters.keys():
                all_clusters[cluster_pred[cell]] = [cell]
            else:
                all_clusters[cluster_pred[cell]].append(cell)

        sys.stdout.write('Found {0} clusters.\n'.format(len(all_clusters.keys())))


        return all_clusters

    def filterNaN(self, snp_input):
        """
        This method cleans up the SNPs that are missing from our input and outputs only the present data structures.

        :param snp_input:
        :return:
        """


        #sys.stdout.write('Filtering missing data...')
        clean_Cell = []
        total_markers = 0
        total_missing = 0
        marker_index = []
        for chrom in range(len(snp_input)):
            #Determine where the missing data is:

            data = np.argwhere(np.isnan(snp_input[chrom][:, 1]) == False).T[0]
            marker_index.append(data) # The indices of the SNPs recorded
            clean_chrom = snp_input[chrom][data]
            clean_Cell.append(clean_chrom)
            total_markers += len(snp_input[chrom][:,1])
            total_missing += (len(snp_input[chrom][:,1]) - len(data))
            #sys.stdout.write('For chromosome {2}: {0} markers missing out of {1} total..\n'.format(len(missing_data), len(snp_input[chrom][:,1]), chroms[chrom]))
        clean_Cell = np.asarray(clean_Cell)

        #sys.stdout.write('{0} markers present out of {1} total...\n'.format(total_markers - total_missing, total_markers))
        marker_out = total_markers - total_missing

        return clean_Cell, marker_out, marker_index

    def computeLikelihoods(self, cluster_IDs):
        """
        Impute the missing genotypes based on cluster calls using likelihood ratio test

        We will apply priors to the model by using the posterior probabilities of the HMM classifications.

        :param cluster_IDs:
        :return:
        """
        str_inds = [str(indiv + 1) for indiv in cluster_IDs]
        sysout = ', '.join(str_inds)
        #sys.stdout.write('Imputing missing SNPs on individuals {0}\n'.format(sysout))

        imputed_genotypes = 0
        for arm in range(len(self.polarized_samples[0])):
            merged_arm = self.polarized_samples[cluster_IDs[0]][arm]
            for individual in range(1, len(cluster_IDs)):
                # Merge the SNP calls from each chromosome arm of each individual
                merged_arm = np.append(merged_arm, self.polarized_samples[cluster_IDs[individual]][arm][:, 1][..., None], 1)

            # Merged all the SNP calls now we will check the consensus of each allele
            arm_calls = np.zeros(shape=(len(merged_arm[:, 0]), 2))
            arm_calls[:, 0] = merged_arm[:, 0]

            snp_index = 0
            for SNP in merged_arm[:, 1:]:
                homozygous_Model = [self.pi_p1_1, self.pi_p1_2, self.pi_p1_3]
                heterozygous_Model = [self.pi_p2_1, self.pi_p2_2, self.pi_p2_3]

                # We will construct a likelihood ratio test between or homozygous and heterozygous model to determine the likelihood of the position given the allele frequency
                L_homo = 0
                L_het = 0
                for call in SNP:
                    # sum log probs
                    if np.isnan(call) == False:  # filter missing data
                        L_homo += np.log(homozygous_Model[int(call)])
                        L_het += np.log(heterozygous_Model[int(call)])
                    else:
                        pass

                #I want to add a routine that would allow me to distinguish between a misclustering -- perhaps by examining the states of the SNPs?

                # Likelihoods for each model are calculated and now can attempt a test
                likelihood = -2 * (L_homo - L_het)
                # Now perform a likelihood ratio and do a hard cutoff at 0 ... this may require a bit more fine tuning later on
                if likelihood == 0:  # Check for NaN arrays and arrays with genotype errors
                    if np.isnan(SNP).all() == True:
                        merged_call = np.nan
                    else:
                        merged_call = 2
                elif likelihood > 0:  # call the site as heterozygous and assign a P2 allele to it
                    merged_call = 1

                elif likelihood < 0:  # call the site as homozygous and assign a P1 allele to it
                    merged_call = 0
                #Made imputation of what SNP will be based on missing data now change it in the data structure
                for individual in range(len(cluster_IDs)):
                    if np.isnan(self.polarized_samples[cluster_IDs[individual]][arm][snp_index][1]) is True or self.polarized_samples[cluster_IDs[individual]][arm][snp_index][1] == 2: #Only impute the genotype if it is missing or has a sequencing error
                        self.polarized_samples[cluster_IDs[individual]][arm][snp_index][1] = merged_call
                    else:
                        pass

                snp_index += 1

    def extractFeatures(self, posterior, full_snp_matrix, data):
        """
        Rather than clustering based on recombination breakpoints which could be more prone to error when I miss classify a segment I am going to cluster
        on a different set of features.

        I am going to use the posterior probability of P1/P1 state at each SNP position as a feature from which to cluster on.
        Since there is missing data I will use smoothing spline interpolation to impute the missing data.

        Now we can return all of this interpolated data and cluster by each of the SNP features.

        :return:
        """
        #Fill out the indices of the SNPs in order for each chromosome

        predict_chr_arms = [[p for p in range(len(full_snp_matrix[arm][:, 0]))] for arm in range(len(full_snp_matrix))]

        ###########################################


        #To make the interpolation behave better we will fill out the ends of our chromosome arrays with a smooth line so we get less weird behavior

        #Fill out posteriors first for 2 and 3
        for arm in range(len(predict_chr_arms)):


            #This dirty little thing extrapolates the posterior probability of P1 for each chromosome arm. This is necessary b/c the edges of the chromosome cannot be interpolated and instead must be extrapolated
            #So we simply take the edge posterior value and extend it until the end of our chromosome. This isn't perfect, but seems to get the job done.

            #iterate through each arm and extrapolate
            right_end = np.asarray([posterior[arm][:, 0] for snp in range(data[arm][0])]).T
            left_end = np.asarray([posterior[arm][:, -1] for snp in range(predict_chr_arms[arm][-1] - data[arm][-1])]).T

            #Concatenate our extrapolated values, and our posterior matrix
            posterior[arm] = np.hstack((right_end, posterior[arm])) if right_end.size else posterior[arm]
            posterior[arm] = np.hstack((posterior[arm], left_end)) if left_end.size else posterior[arm]

            # now fill out the x-axis
            right_end_snp = [snp for snp in range(data[arm][0])]
            left_end_snp = [snp + predict_chr_arms[arm][-1] for snp in range(predict_chr_arms[arm][-1] - data[arm][-1])]
            data[arm] = np.hstack((right_end_snp, data[arm], left_end_snp))

        #######
        training_chr_arms = data #Use the SNP indices as the x-axis and not the position along the chromosome as the interpolation behaves better this way





        #Use interpolation to fill out the missing values
        #This will allow us to impute the posterior probs of our missing SNPs

        imputed_probs = []
        for chrom in range(len(data)):
            posterior_prob = np.exp(posterior[chrom][0])

            #Method for implementing a linear interpolation
            predict_probs = np.interp(x=predict_chr_arms[chrom], xp=training_chr_arms[chrom], fp=posterior_prob)

            imputed_probs = np.hstack((imputed_probs, predict_probs))

        return np.asarray(imputed_probs)

    def paintChromosome(self, breakpoints, labels):
        """
        This method will "paint" the genome with our homozygous or heterozygous labels from our HMM classifier and impute the missing data for us.
        We will use this method to pileup our genotypes from each individual and infer the final allele frequencies

        :param breakpoints:
        :param labels:
        :return:
        """

        painted_chromosomes = []
        for chrom in range(len(self.paintedGenome)):

            prev_chiasma = 0
            paintedSegment = []
            if len(breakpoints[chrom]) != 0: #If a breakpoint is called at all then we paint the chromosome with different labels
                for chiasma in range(len(breakpoints[chrom])):
                    #For each chromosome we have a single tuple for each chiasma -- every breakpoint has an object that will declare a change from P1 -> P2 or vice versa
                    if chiasma != len(breakpoints[chrom])-1: #For all segments except the last segment take an intersection between the previous breakpoint and the current one
                        segment_A = self.paintedGenome[chrom][np.where(self.paintedGenome[chrom] < breakpoints[chrom][chiasma])]
                        segment_B = self.paintedGenome[chrom][np.where(self.paintedGenome[chrom] >= prev_chiasma)]
                        label_segment = np.intersect1d(segment_A, segment_B)
                        paint = np.full((len(label_segment)), labels[chrom][chiasma][0])

                    else: #If it is the last chiasma we must get the piece before the breakpoint and the piece after the breakpoint
                        segment_A = self.paintedGenome[chrom][np.where(self.paintedGenome[chrom] < breakpoints[chrom][chiasma])]
                        segment_B = self.paintedGenome[chrom][np.where(self.paintedGenome[chrom] >= prev_chiasma)]
                        label_segment = np.intersect1d(segment_A, segment_B)
                        mid_paint = np.full((len(label_segment)), labels[chrom][chiasma][0])

                        end_segment = self.paintedGenome[chrom][np.where(self.paintedGenome[chrom] >= breakpoints[chrom][chiasma])]
                        end_paint = np.full((len(end_segment)), labels[chrom][chiasma][1])
                        paint = np.concatenate((mid_paint, end_paint))

                    #Simple method to account concatenate labelling
                    if len(paintedSegment) == 0:
                        paintedSegment = paint
                    else:
                        paintedSegment = np.concatenate((paintedSegment, paint))


                    prev_chiasma = breakpoints[chrom][chiasma]
            else: # we simply call the chromosome by a single label
                paintedSegment = np.full((len(self.paintedGenome[chrom])), labels[chrom])

            painted_chromosomes.append(paintedSegment)

        return painted_chromosomes

    def cM_map(self, arm):
        """
        The function to map a bp coordinate to cM coordinate
        Use this function to create maps

        :param arm:
        :return:
        """


        cM_bins = self.rmap.loc[self.rmap[0] == self.chroms[arm]].values[:,1:4] #Get the bins for the selected chromosome/arm
        get_cM = lambda x: cM_bins[np.intersect1d(np.where(x >= cM_bins[:,1]), np.where(x < cM_bins[:,2]))][0][0]


        return get_cM

    def expect_AF(self, cM, D, drive_locus):
        """
        This function will be applied to a vector of positions in cM in order to compute the expected allele frequency at those positions. Using this function I will be able to do curve fitting to estimate
        the strength of drive and the drive_locus position.
        """

        P_drive = .5 + D
        P_odd = (lambda x: (1 - math.exp(-((2 * x) / 100))) / 2)
        E_p1 = (lambda x: .5 * (P_drive * (1 - x) + (1 - P_drive) * x) + .5)
        distance = abs(cM - drive_locus)
        CO_prob = np.array(list(map(P_odd, distance)))
        E_AF = 1 - np.array(list(map(E_p1, CO_prob)))

        return E_AF

    def read_heterochromatin(self):
        """
        This function will use the borders of the recombination map bed file supplied and build the heterochromatin boundaries.

        To do this we have to find where the ends of the recombination map are and also determine which region is the centromeric
        non-recombining heterochromatin which would have gap in cM distance.
        :return:
        """
        c_index = 0
        for chrom in self.chroms:
            cM_bins = self.rmap.loc[self.rmap[0] == chrom].values[:,1:4]  # Get the bins for the selected chromosome/arm


            #Then we must scan across the array and look for the centromere which should be detectable by looking for non-uniform jumps in cM
            D_vector = np.full(fill_value=np.nan, shape=cM_bins.shape[0]-1)
            for cM in range(1, cM_bins.shape[0]):
                D = abs(cM_bins[:,0][cM-1] - cM_bins[:,0][cM])
                D_vector[cM-1] = D
            if len(cM_bins[np.where(D_vector > 1)]) > 0:
                self.heterochromatin[c_index] =[ [cM_bins[0][1], cM_bins[:,2][np.where(D_vector > 1)][0] ], [cM_bins[:,1][np.where(D_vector > 1)[0] + 1][0], cM_bins[-1][2]] ]#now we get the borders of our heterochromatin boundaries
                self.cM_boundaries[c_index] = [ [cM_bins[0][0], cM_bins[:,0][np.where(D_vector > 1)][0] ], [cM_bins[:,0][np.where(D_vector > 1)[0] + 1][0], cM_bins[-1][0]] ]
            else: #The chromosome could have no centromere in which case we do not want to call that region
                self.heterochromatin[c_index] = [[cM_bins[0][1], cM_bins[-1][2]]]
                self.cM_boundaries[c_index] = [[cM_bins[0][0], cM_bins[-1][0]]]
            c_index += 1

    def estimateSD_params(self, AF_data):
        """
        This method uses curve fitting least squares non-linear regression to fit my observed AF data to a model of segregation distortion.
        I will use this curve fitting to estimate both the strength and location of my segregation distortion parameter.

        :return:
        """

        sys.stdout.write('Fitting parameters for segregation distortion inference...\n')

        transformed_DIST = []

        for arm in range(len(self.paintedGenome)):
            #Check the output of this function to make sure it is correct
            arm_pos_AF = []

            for euchromatin in self.heterochromatin[arm]:

                if len(arm_pos_AF) == 0:
                    arm_pos_AF = AF_data[arm][np.intersect1d(np.where(AF_data[arm][:,0] < euchromatin[1]), np.where(AF_data[arm][:,0] > euchromatin[0]))]
                else:
                    arm_pos_AF = np.concatenate((arm_pos_AF, AF_data[arm][np.intersect1d(np.where(AF_data[arm][:,0] < euchromatin[1]), np.where(AF_data[arm][:,0] > euchromatin[0]))]))
            cM_pos = np.array(list(map(self.cM_map(arm), arm_pos_AF[:,0]))) #convert Bp to cM
            cM_AF = np.hstack((cM_pos.reshape(-1,1), arm_pos_AF))
            transformed_DIST.append(cM_AF)


        cM_genome = transformed_DIST

        fitted_allele_frequencies = []
        param_estimates = []
        param_errs = []
        for chrom in range(len(cM_genome)):

            #As a boundary for the drive locus parameter we are going to limit to the centromere

            if len(self.heterochromatin[chrom]) == 1:
                #Telocentric arm (chrX dMel)
                het_boundary = [self.cM_boundaries[chrom][0][1]-2, self.cM_boundaries[chrom][0][1]]
            else:
                #metacentric (Chr2, Chr3 dMel) this function will not quite be correct for Dvir b/c their centromeres are in different coordinate space
                het_boundary = [self.cM_boundaries[chrom][0][1], self.cM_boundaries[chrom][1][0]]

            #Determine an error term for our allele frequencies:
            #in the future this will be more well defined, but for now we can just use the adjusted sample size

            SD_params, param_cov = curve_fit(self.expect_AF, cM_genome[chrom][:,0], cM_genome[chrom][:, 2], bounds=([-5, het_boundary[0]],[.5, het_boundary[1]])) #Predict the parameters of our model which are the strength of drive and the position of drive locus
            param_estimates.append(SD_params)
            param_errs.append(np.sqrt(np.diag(param_cov)))

            predAF = self.expect_AF(cM_genome[chrom][:,0], D = SD_params[0], drive_locus=SD_params[1]) #Use parameter estimates to output predict AF for our positions
            fitted_allele_frequencies.append(np.column_stack((cM_genome[chrom][:,1], predAF)) )

        return param_estimates, param_errs, fitted_allele_frequencies

    def pseudoBulk(self, all_individuals):


        """
        This function will compute the pseudobulk data from our single cell SNP reads.

        In order to reduce noise I will bin the genome into 200kb bins as demonstrated in Wei et al., 2017

        :param all_individuals:
        :return:
        """

        sys.stdout.write('Calculating allele frequencies for pseudobulk data in 200kb bins\n')
        SNP_positions= [self.polarized_samples[0][arm][:, 0] for arm in range(len(self.polarized_samples[0]))]

        SNP_pileup = [[] for p in range(len(self.polarized_samples[0]))]

        #pile up all of the SNPs into a matrix
        for label in all_individuals:
            cell = self.polarized_samples[label]
            for arm in range(len(cell)):
                if len(SNP_pileup[arm]) == 0:
                    SNP_pileup[arm] = cell[arm][:, 1]
                else:
                    SNP_pileup[arm] = np.vstack((SNP_pileup[arm], cell[arm][:, 1]))

        for arm in range(len(self.polarized_samples[0])):
            SNP_pileup[arm] = SNP_pileup[arm].T
            # We will smooth out the SNP pileups by binning the genome in 200kb bins
            midPoints = np.asarray([(step+100000)/1000000 for step in range(int(min(SNP_positions[arm])), int(max(SNP_positions[arm]))+200000, 200000)])

            AF_bins = np.zeros(shape=(len(midPoints), 2))
            AF_bins[:,0] = midPoints

            position = 0
            for step in range(int(min(SNP_positions[arm])), int(max(SNP_positions[arm]))+200000, 200000): #Iterate from min SNP to max SNP in 200kb steps

                bin_snps = np.intersect1d(np.where(SNP_positions[arm] >= step), np.where(SNP_positions[arm] < step + 200000)) # Get the indices where this bin resides

                if len(bin_snps) == 0:#if there are no SNPs in the given bin then pass the allele frequency from the bin - 1 to this bin for simplicity
                    AF_bins[:, 1][position] = AF_bins[:, 1][position-1]
                else:
                    tot_SNPs = abs(np.sum(np.isnan(SNP_pileup[arm][bin_snps]).astype(int), axis=1) - len(all_individuals)) #Total number of non-NaN markers (# of NaN markers - total individuals for each SNP)
                    SNP_freq = np.nansum(SNP_pileup[arm][bin_snps], axis=1) / tot_SNPs #Divide the freq of P2 allele over the number of non-NaN markers
                    bin_freq = np.average(SNP_freq) #Compute the average AF in the bin
                    AF_bins[:,1][position] = bin_freq
                position += 1


            self.pseudoBulk_data.append(AF_bins)

def wrapProcess():
    """
    Wraps the multiprocessing of the posterior probability inference

    :return:
    """
    #Data structures

    snp_len = 0
    for length in range(len(myAnalysis.polarized_samples[0])):
        snp_len += len(myAnalysis.polarized_samples[0][length][:,0])
    all_features = np.zeros(shape=(len(myAnalysis.polarized_samples), snp_len))

    # Run the functions
    sys.stdout.write('Constructing posterior probability matrices...\n')
    myPool = Pool(processes=myArgs.args.threads)
    ML_result = myPool.map(multiThreaded, myAnalysis.polarized_samples)
    myPool.close()


    for pred in range(len(ML_result)):
        all_features[pred] = ML_result[pred]

    path, file = os.path.split(myArgs.args.snp)
    out_file = file.split('.')[0] + '_HMM_posteriors.npy'
    output = os.path.join(path, out_file)
    np.save(output, all_features)
    #sys.exit()

    # Cluster method:
    clusters = myAnalysis.cluster_RBPs(predictions=all_features, distance=myArgs.args.distance, threads=myArgs.args.threads)
    delta_indiv = mergeIndividuals(clusters, threads=myArgs.args.threads)

    return delta_indiv, all_features, clusters

def mergeIndividuals(clusters, threads):
    """
    After clusters are called we will merge the SNP inputs of the individuals that had clusters with more than one individual.
    This will fill out some of the missing data and will increase our certainty on the breakpoint predictions when we re-run our HMM and see if we get a change in clustering.


    :return:
    """


    sys.stdout.write('Initializing genotype imputation...\n')

    mergeable_clusters = [clusters[mc] for mc in clusters.keys() if len(clusters[mc]) > 1]

    if len(mergeable_clusters) > 0:
        ###################################################################
        # To speed up the process I have implemented multithreading
        myPool = Pool(processes=threads)
        myPool.map(wrapComputeLikelihdoods, mergeable_clusters)
        myPool.close()
    ##################################################################

    return len(clusters.keys())

def wrapComputeLikelihdoods(cluster_labels):
    #Really stupid simply function that needs to be written to wrap the computeLikelihood function so that the pickling in the multiprocessing module will work correctly
    #There are problems when using lambda functions in <__init__> function and the class doesn't get pickled, but this should avoid pickling the class

    myAnalysis.computeLikelihoods(cluster_IDs=cluster_labels)

def multiThreaded(cell_snp):

    sub_sampling, markers, SNP_index = myAnalysis.filterNaN(snp_input=cell_snp)
    posterior_probs = myAnalysis.hmmFwBw(sub_sampling)
    SNP_features = myAnalysis.extractFeatures(posterior_probs, cell_snp, SNP_index)


    return SNP_features

def imputeSegments(cell):
    """
    After we have converged on cluster labels we pass to this method so that we can do the final classification of the data and then do the final imputation of all missing data.

    This will be done by using the viterbi decoding to determine the maximum likelihood path through the data. We will using this to call the recombination breakpoints indpendently
    in each individual of every cluster first. This could give us the ability to construct an interval of prediction recmobination breakpoints. Then we will compute this for the rest of the individuals.

    The posterior probability will also be used to construct an interval around our predicted breakpoint so that we can display some uncertainty in our prediction in the same way that I initial attempted
    to infer the recombination breakpoints for the clustering.

    We are still subject to the problem of a failed prediction of a recombination breakpoint. Hopefully, after clustering many of the individuals this problem will be less problematic as there will be more data
    imputed.

    :param cluster_labels:
    :return:
    """


    sub_sampling, markers, SNP_index = myAnalysis.filterNaN(myAnalysis.polarized_samples[cell])
    cell_pointers = myAnalysis.hmmViterbi(snp_input=sub_sampling)
    ML_predicts, ML_labels = myAnalysis.decode(cell_pointers, sub_sampling)
    imputedSNPs = myAnalysis.paintChromosome(breakpoints=ML_predicts, labels=ML_labels)


    return imputedSNPs

def multImpute(cluster_labels):


    sys.stdout.write('Implementing maximum likelihood inference of P1 and P1/P2 states...\n')

    myAnalysis.paintedGenome = [myAnalysis.polarized_samples[0][arm][:, 0] / 1000000 for arm in range(len(myAnalysis.polarized_samples[0]))]

    #Get the HMM inferred parameter allele frequencies for the collapsed clusters:
    collapsed_individuals = [random.choice(cluster_labels[clust]) for clust in cluster_labels.keys()]
    myPool = Pool(processes=myArgs.args.threads)
    SNP_pileup = myPool.map(imputeSegments, collapsed_individuals)
    myPool.close()




    sys.stdout.write('Inferring final allele frequency for population...\n')
    for arm in range(len(myAnalysis.paintedGenome)):
        myAnalysis.paintedGenome[arm] = np.vstack((myAnalysis.paintedGenome[arm], np.sum(np.vstack(np.vstack(SNP_pileup)[:,arm]), axis=0) / (2*len(collapsed_individuals)))).T

    #Perform parameter estimation on all of my inferred allele frequencies

    HMM_P, HMM_err, HMM_fitted_allele = myAnalysis.estimateSD_params(AF_data=myAnalysis.paintedGenome)
    pseudo_P, pseudo_Err, pseudo_fitted_allele = myAnalysis.estimateSD_params(AF_data=myAnalysis.pseudoBulk_data)



    path, file = os.path.split(myArgs.args.snp)

    plot(HMM_fitted_allele, HMM_P, HMM_err, pseudo_P, pseudo_Err, pseudo_fitted_allele, name='{0}.imputed_AF_avg_rate'.format(file.split('.')[0]))

def plot(HMM_fit, HMM_P, HMM_err, pseudo_P, pseudo_Err, pseudo_fit, name='test', ):



    with sns.axes_style('whitegrid'):
        fig = plt.figure(figsize=(15, 10))
        axs = fig.subplots(len(myAnalysis.chroms))
        for p in range(len(myAnalysis.chroms)):
            sns.scatterplot(myAnalysis.paintedGenome[p][:, 0], myAnalysis.paintedGenome[p][:, 1], ax=axs[p], edgecolor=None, alpha=.1, color='black')
            sns.scatterplot(myAnalysis.pseudoBulk_data[p][:,0], myAnalysis.pseudoBulk_data[p][:,1], ax=axs[p], edgecolor=None, alpha=.5, color='red')
            axs[p].plot(HMM_fit[p][:,0], HMM_fit[p][:,1], linestyle='--', color='blue')
            axs[p].plot(pseudo_fit[p][:,0], pseudo_fit[p][:,1], linestyle='--', color='green')
            axs[p].set_title(myAnalysis.chroms[p])
            axs[p].set_ylabel('P2 AF')
            axs[p].text(x=0.5, y=-0.4, s='HMM: Driver strength: {0:.2f} +/- {2:.2f}% Drive locus: {1:.0f} cM\n Pseudobulk: Driver strength: {3:.2f} +/- {4:.2f}% Drive locus: {5:.0f} cM'.format(
                HMM_P[p][0]*100, HMM_P[p][1], HMM_err[p][0]*100,
                pseudo_P[p][0]*100, pseudo_Err[p][0]*100, pseudo_P[p][1]), size=10,
                ha='center', transform=axs[p].transAxes)
    fig.legend(['HMM AF', 'Pseudobulk AF', 'HMM fitted curve', 'Pseudobulk fitted curve'])
    axs[2].set_xlabel('Mb')
    plt.tight_layout(h_pad=1)
    #plt.show()
    file_name = os.path.join('/home/iskander/Documents/Barbash_lab/mDrive/AF_PLOTS', name)
    plt.savefig('{0}.png'.format(file_name), dpi=300)
    plt.close()

if __name__ == '__main__':
    start = time.time()
    myArgs = CommandLine()
    if myArgs.args.polarize == True:#If program is called to polarize the SNPs in preprocessing
        sys.stdout.write('You made a mistake. This option does nothing.')
    else:
        sys.stdout.write('Initializing single cell analysis...\nSNP input:{0}\n'.format(myArgs.args.snp))

        #### Multithreaded ####
        sys.stdout.write('{0} thread(s) to be used.\n'.format(myArgs.args.threads))
        snp_path, snp_file = os.path.split(myArgs.args.snp)

        myAnalysis = analyzeSEQ()
        myAnalysis.polarized_samples = myAnalysis.load_SNP_array(path=snp_path, snp_array=snp_file, encoding='latin1')

        #Instantiate variables
        myAnalysis.recomb_rate = myArgs.args.rate
        myAnalysis.rmap = pd.read_csv(myArgs.args.rmap, header=None, sep='\t')


        seen_chroms = set()
        add_chroms = seen_chroms.add
        myAnalysis.chroms = [x for x in myAnalysis.rmap.values[:, 0] if not (x in seen_chroms or add_chroms(x))] #add the chromosome names in order from the rmap file
        myAnalysis.read_heterochromatin()

        #Psuedobulk
        myAnalysis.pseudoBulk([i for i in range(len(myAnalysis.polarized_samples))])#Compute pseuddbulk before genotype imputation

        ### Run HMM, Cluster ###
        for i in range(1):
            cluster_num, posteriors, cluster_labels = wrapProcess()

        ### Calculate the allele frequencies via HMM and est. SD ###
        multImpute(cluster_labels)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    sys.stdout.write("Time elapsed: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
