import os
import numpy as np
import csv
import operator
import math
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import scipy.stats as stats

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import linkage, fcluster, fclusterdata
import argparse

"""
------------------------------------------------------------------------------------
MIT License

Copyright (c) 2018 Iskander Said

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
                                 default='/home/iskander/Documents/MEIOTIC_DRIVE/polarized_simulations.npy')
        self.parser.add_argument("-r", "--reference", type=str, action="store", nargs="?",
                                 help="The reference genome numpy file to polarize SNPs",
                                 default='/home/iskander/Documents/MEIOTIC_DRIVE/882_129.snp_reference.npy')

        if inOpts is None:  ## DONT REMOVE THIS. I DONT KNOW WHAT IT DOES BUT IT BREAKS IT
            self.args = self.parser.parse_args()
        else:
            self.args = self.parser.parse_args(inOpts)



"""

A program for analyzing single cell data.

"""


class analyzeSEQ:

    def __init__(self):
        self.polarized_samples = []

        ####Training data####
        self.homozygous_segments = np.zeros(shape=(1, 3))
        self.heterozygous_segments = np.zeros(shape=(1, 3))


        ##### Set parameters for the multinomials ###

        #P1 homozygous
        self.pi_p1_1 = 0.9993 #P1 allele in P1 homozygoous region
        self.pi_p1_2 = 0.0002 #P2 allele in P1 homozygous region
        self.pi_p1_3 = 0.0005 #Seq error in P1 homozygous region

        self.pi_p2_1 = 0.5005 #P1 allele in P1/P2
        self.pi_p2_2 = 0.499 #P2 allele in P1/P2
        self.pi_p2_3 = 0.0005 #Seq error in P1/P2



        ###Probability states for the HMM###

        self.init_chrom_states = {'p1':np.log(0.5), 'p2':np.log(0.5)}
        self.state_probs = {'p1':[np.log(self.pi_p1_1), np.log(self.pi_p1_2), np.log(self.pi_p1_3)],
                       'p2': [np.log(self.pi_p2_1), np.log(self.pi_p2_2), np.log(self.pi_p2_3)]
                       }
        self.transitions_probs ={'p1':{'p1':np.log(1 - self.simple_cM()), 'p2':np.log(self.simple_cM())}, 'p2':{'p1': np.log(self.simple_cM()), 'p2':np.log(1-self.simple_cM())}}

        # manually tune some parameters:

        # myAnalysis.pi_p1_1 = .92
        # myAnalysis.pi_p1_2 = .075
        # myAnalysis.pi_p1_3 = .005

        # myAnalysis.pi_p2_1 = .56
        # myAnalysis.pi_p2_2 = .435
        # myAnalysis.pi_p2_3 = .005

    def get_cM(self, chromosome_arm, mid_point, distance, heterochromatin=((0.53, 18.87),(1.87, 20.87),(0.75, 19.02), (2.58, 27.44),(1.22, 21.21))):
        """
        Calculate the transition probabilty as a function of recombination distance between each SNP marker in array of emissions as calculated by
        https://petrov.stanford.edu/cgi-bin/recombination-rates_updateR5.pl#Method


        :param chromosome_arm:
        :param mid_point:
        :param distance:
        :param heterochromatin:
        :return:
        """



        if chromosome_arm == 0: #2L
            if mid_point >= heterochromatin[0][0] and mid_point <= heterochromatin[0][1]:
                rate = 0
            else:
                rate = 2.58909 + 0.40558 * mid_point - 0.02886 * mid_point** 2
        elif chromosome_arm == 1: #2R
            if mid_point >= heterochromatin[1][0] and mid_point <= heterochromatin[1][1]:
                rate = 0
            else:
                rate =  -1.435345 + 0.698356 * mid_point - 0.023364 * mid_point**2
        elif chromosome_arm == 2: #2L
            if mid_point >= heterochromatin[2][0] and mid_point <= heterochromatin[2][1]:
                rate = 0
            else:
                pass

    def simple_cM(self, dist=100):
        """
        A simple calculation for transition probability using the genome wide average cM/Mb rate: 2.46 cM/Mb

        :param dist:
        :return:
        """

        trans_prob = 1 - math.exp(-(dist * 2.46*10**-8))
        return trans_prob

    def decode(self, pointers, snp_matrix):

        rbp_positions= np.zeros(shape=(1,3)) #chr2, chr3, chrx
        rbp_indices = np.zeros(shape=(1,3))

        chr_2 = np.concatenate((snp_matrix[0][:, 0], snp_matrix[1][:, 0] + 23000000)) / 1000000
        chr_3 = np.concatenate((snp_matrix[2][:, 0], snp_matrix[3][:, 0] + 24500000)) / 1000000
        chr_x = snp_matrix[4][:,0] / 1000000

        all_chr_arms = [chr_2, chr_3, chr_x]
        chr_index = 0
        for traceback in pointers:
            #iterate through pointers for each chromosome
            for position in range(1, len(traceback)):
                #Check for a change in state from P1/P1 to P1/P2
                if traceback[position-1] != traceback[position]:
                    rbp_positions[0][chr_index] = all_chr_arms[chr_index][position]
                    rbp_indices[0][chr_index] = position

            chr_index += 1

        return rbp_positions, rbp_indices
    def hmmFwBw(self, snp_input):
        states = ['p1', 'p2']
        chr_posteriors = []

        chr_2 = np.concatenate((snp_input[0][:, 0], snp_input[1][:, 0] + 23000000))
        chr_3 = np.concatenate((snp_input[2][:, 0], snp_input[3][:, 0] + 24500000))
        chr_x = snp_input[4][:, 0]
        all_chr_arms = [chr_2, False, chr_3, False,
                        chr_x]  # Have to put these placeholders in this because I have developed a weird codebase and I reap what I so


        # Fwd Bckwd

        for chrom in range(0, 5, 2):  # consider each chromosome as a single sequence
            if chrom != 4:
                chrom_length = len(snp_input[chrom]) + len(snp_input[chrom + 1])
            else:
                chrom_length = len(snp_input[chrom])

            fMatrix = np.zeros(shape=(2, chrom_length))


            # First state
            # P(state) * P(emission|state)
            init_emission = np.nonzero(snp_input[chrom][0][1:4])[0][0]

            for state in range(2):
                fMatrix[state][0] = self.init_chrom_states[states[state]] + self.state_probs[states[state]][
                    init_emission]


            # Proceed along sequence of SNPs and compute probabilities
            if chrom != 4:
                chromosome_arm = np.vstack((snp_input[chrom], snp_input[chrom + 1]))
            else:
                chromosome_arm = snp_input[chrom]

            for SNP in range(1, len(chromosome_arm)):
                dist = abs(all_chr_arms[chrom][SNP] - all_chr_arms[chrom][SNP - 1])

                # Reinstantiate the transition probabilities with distance parameter at each iteration
                self.transitions_probs = {
                    'p1': {'p1': np.log(1 - self.simple_cM(dist)), 'p2': np.log(self.simple_cM(dist))},
                    'p2': {'p1': np.log(self.simple_cM(dist)), 'p2': np.log(1 - self.simple_cM(dist))}}


                emission = np.nonzero(chromosome_arm[SNP][1:4])[0][0]
                for state in range(2):
                    # P(emission|state) * Sum(P(transition to state) * P(state -1), P(transition to state) * P( state -1))
                    sumScore = self.sumLogProb(self.transitions_probs['p1'][states[state]] + fMatrix[0][SNP-1],
                                               self.transitions_probs['p2'][states[state]] + fMatrix[1][SNP-1])
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

                dist = abs(all_chr_arms[chrom][SNP] - all_chr_arms[chrom][SNP + 1])

                # Reinstantiate the transition probabilities with distance parameter at each iteration
                self.transitions_probs = {
                    'p1': {'p1': np.log(1 - self.simple_cM(dist)), 'p2': np.log(self.simple_cM(dist))},
                    'p2': {'p1': np.log(self.simple_cM(dist)), 'p2': np.log(1 - self.simple_cM(dist))}}

                # First loop is to go through sequence
                for state in range(2):
                    # Sum(P(transition to state) * P(state + 1) * P(state +1),
                    #  P(transition to state) * P( state + 1) * P(emission + 1|state+1) )
                    emission = np.nonzero(chromosome_arm[SNP+1][1:4])[0][0]

                    bestScore = self.sumLogProb(
                        self.state_probs['p1'][emission] + self.transitions_probs['p1'][states[state]] + bMatrix[0][SNP + 1],

                        self.state_probs['p2'][emission] + self.transitions_probs['p2'][states[state]] + bMatrix[1][SNP + 1])

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



        return chr_posteriors
    def draw(self, posterior_matrix, snp_matrix, predicted_rbps, truth, title='TestID'):
        chr_decoding = {0:'Chr2', 1:'Chr3', 2:'ChrX'}

        chr_2 = np.concatenate((snp_matrix[0][:, 0], snp_matrix[1][:, 0] + 23000000)) / 1000000
        chr_3 = np.concatenate((snp_matrix[2][:, 0], snp_matrix[3][:, 0] + 24500000)) / 1000000
        chr_x = snp_matrix[4][:, 0] / 1000000
        all_chr_arms = [chr_2, chr_3, chr_x]


        chr2_alleles = [np.nonzero(snp)[0][0] for snp in np.concatenate((snp_matrix[0][ :,[1,2,3] ], snp_matrix[1][ :,[1,2,3] ]))]
        chr3_alleles = [np.nonzero(snp)[0][0] for snp in np.concatenate((snp_matrix[2][ :,[1,2,3] ], snp_matrix[3][ :,[1,2,3] ]))]
        chrX_alleles = [np.nonzero(snp)[0][0] for snp in snp_matrix[4][ :,[1,2,3] ]]
        all_alleles = [chr2_alleles, chr3_alleles, chrX_alleles]
        with sns.axes_style('darkgrid'):
            fig = plt.figure(figsize=(20, 10))
            axs = fig.subplots(3,2)


        for arm in range(3):

            sns.lineplot(all_chr_arms[arm], all_alleles[arm], ax=axs[arm][1])
            sns.lineplot(all_chr_arms[arm], posterior_matrix[arm][0], ax=axs[arm][0])
            sns.lineplot(all_chr_arms[arm], posterior_matrix[arm][1], ax=axs[arm][0])
            axs[arm][0].set_title('{0}'.format(chr_decoding[arm]))
            axs[arm][0].set_ylabel('log(probability)')
            axs[arm][0].set_xlabel('Position (Mb)')


            axs[arm][1].axvline(truth[arm], linestyle='--', color='blue')
            axs[arm][1].set_yticks([0,1,2])
            axs[arm][1].set_yticklabels(['P1', 'P2', 'Error'])
            axs[arm][0].axvline(predicted_rbps[arm], linestyle='--', color='red')
            axs[arm][0].axvline(truth[arm], linestyle='--', color='blue')
            other_legend = ['Parental alleles', 'True Breakpoint: {0:.0f} bp'.format(truth[arm] * 1000000)]
            legend = ['P1/P1 Probability', 'P1/P2 Probability','Predicted Breakpoint: {0:.0f} bp'.format(predicted_rbps[arm] * 1000000), 'True Breakpoint: {0:.0f} bp'.format(truth[arm] * 1000000)]
            axs[arm][0].legend(legend)
            axs[arm][1].legend(other_legend)


        plt.suptitle('Recombination Breakpoint Predictions'.format(title))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_name = os.path.join('/home/iskander/Documents/MEIOTIC_DRIVE/', '{0}.png'.format(title))
        plt.savefig(plot_name, dpi=200)
        #plt.show()
    def hmmViterbi(self, snp_input):

        """
        This function will perform our viterbi decoding for our recombination breakpoint detection. Depending on how fuzzy the breakpoint positions end up looking
        it may just be better to use the posterior distribution of the states and take like an area around the inflection point of the changes in posterior probabilities.


        :param snp_input:
        :return:
        """

        states = ['p1', 'p2']
        arm_pointers = []


        chr_2 = np.concatenate((snp_input[0][:, 0], snp_input[1][:, 0] + 23000000))
        chr_3 = np.concatenate((snp_input[2][:, 0], snp_input[3][:, 0] + 24500000))
        chr_x = snp_input[4][:, 0]
        all_chr_arms = [chr_2, False, chr_3, False, chr_x] #Have to put these placeholders in this because I have developed a weird codebase and I reap what I sow


        #Viterbi algorithm
        for chrom in range(0, 5, 2):# consider each chromosome as a single sequence
            if chrom != 4:
                chrom_length = len(snp_input[chrom]) + len(snp_input[chrom+1])
            else:
                chrom_length = len(snp_input[chrom])

            hmmMatrix = np.zeros(shape=(2, chrom_length))
            tracebackMatrix = np.zeros(shape=(2, chrom_length))

            #First state
            # P(state) * P(emission|state)
            init_emission = np.nonzero(snp_input[chrom][0][1:4])[0][0]

            for state in range(2):
                hmmMatrix[state][0] = self.init_chrom_states[states[state]] + self.state_probs[states[state]][init_emission]
                tracebackMatrix[state][0] = state


            #Proceed along sequence of SNPs and compute probabilities and fill traceback matrix
            if chrom != 4:
                chromosome_arm = np.vstack((snp_input[chrom], snp_input[chrom+1]))
            else:
                chromosome_arm = snp_input[chrom]

            for SNP in range(1, len(chromosome_arm)):

                dist = abs(all_chr_arms[chrom][SNP] - all_chr_arms[chrom][SNP-1])

                #Reinstantiate the transition probabilities with distance parameter at each iteration
                self.transitions_probs = {'p1': {'p1': np.log(1 - self.simple_cM(dist)), 'p2': np.log(self.simple_cM(dist))},
                                          'p2': {'p1': np.log(self.simple_cM(dist)), 'p2': np.log(1 - self.simple_cM(dist))}}
                emission = np.nonzero(chromosome_arm[SNP][1:4])[0][0]
                for state in range(2):


                    # P(emission|state) * Max (P(transition to state) * P(state -1), P(transition to state) * P( state -1)
                    bestScore = max((self.transitions_probs['p1'][states[state]] + hmmMatrix[0][SNP - 1], 0),
                                    (self.transitions_probs['p2'][states[state]] + hmmMatrix[1][SNP - 1], 1),
                                    key=operator.itemgetter(0))

                    newScore = (self.state_probs[states[state]][emission] + bestScore[0], bestScore[1])
                    hmmMatrix[state][SNP] = newScore[0]
                    tracebackMatrix[state][SNP] = newScore[1]

            # Traceback matrix and probability matrix have been instantiated now we will use traceback matrix to point back and call hidden states
            startPos = int(max(zip(hmmMatrix[:, -1], tracebackMatrix[:, -1]))[1])
            # print(max(zip(hmmMatrix[:,-1],tracebackMatrix[:,-1]))[0])
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
    def supervisedTraining(self, crossovers, path):
        """
        Use my simulated data to infer parameters of my multinomial distributions given that I know which regions will be homozygous
        or heterozygous.

        :param snp_input:
        :param crossovers:
        :return:
        """

        chr_key = {'2L' : 0, '2R': 1, '3L':2, '3R':3, 'X':4}

        crossover_set = os.path.join(path, crossovers)
        with open(crossover_set, 'r') as myCO:
            coRead = csv.reader(myCO, delimiter='\t')
            next(coRead)
            skip_count = 0
            training_set = []
            for field in coRead:
                if skip_count % 2 == 0:
        ### Determine the homozygous and heterozygous segments and pull out relevant SNPs####
                    sampleIndex = int(field[0]) - 1
                    chrIndex_2 = [chr_key[field[1]] , int(field[2]), int(field[3]) ]
                    chrIndex_3 = [chr_key[field[4]], int(field[5]), int(field[6]) ]
                    chrIndex_X = [chr_key[field[7]], int(field[8]), int(field[9]) ]

                    self.extractSegments(chrIndex_2, chrIndex_3, chrIndex_X, sampleIndex, subSample=.05)

                else:
                    pass
                skip_count += 1

            myCO.close()


        pi_P1 = self.homozygous_segments / np.sum(self.homozygous_segments)
        pi_P2 = self.heterozygous_segments / np.sum(self.heterozygous_segments)

        return pi_P1, pi_P2
    def extractSegments(self, chr2, chr3, chrX, sampleID, subSample=False):
        if subSample != False:#If you call subsampling
            cov = np.random.exponential(subSample)
            snp_array = self.subSample_SNP(self.polarized_samples[sampleID], sampling=cov)#Sub sample singular cell
            x_snps = snp_array[chrX[0]]
            two_snps = snp_array[chr2[0]]
            three_snps = snp_array[chr3[0]]

        else:
            x_snps = self.polarized_samples[sampleID][chrX[0]]
            two_snps = self.polarized_samples[sampleID][chr2[0]]
            three_snps = self.polarized_samples[sampleID][chr3[0]]

        ###Extract relevant SNP counts for ChrX####
        #x_snps = self.polarized_samples[sampleID][chrX[0]]
        x_bp = chrX[1]
        if chrX[2] == 1: #Gamete is P1, P2
            for position in range(len(x_snps)):
                if x_snps[position][0] >= x_bp:

                    self.heterozygous_segments = self.heterozygous_segments + np.sum(x_snps[position:][:,[1,2,3]], axis=0)
                    self.homozygous_segments = self.homozygous_segments + np.sum(x_snps[0:position][:,[1,2,3]], axis=0)

                    break
        else: # gamete was P2, P1
            for position in range(len(x_snps)):
                if x_snps[position][0] >= x_bp:

                    self.homozygous_segments = self.homozygous_segments + np.sum(x_snps[position:][:,[1,2,3]], axis=0)
                    self.heterozygous_segments = self.heterozygous_segments + np.sum(x_snps[0:position][:,[1,2,3]],axis=0)
                    break


        ### Extract relevant SNP counts for Chr2 ##
        #two_snps = self.polarized_samples[sampleID][chr2[0]]
        two_bp = chr2[1]
        if chr2[2] == 1: #P1, P2

            for position in range(len(two_snps)):
                if two_snps[position][0] >= two_bp:
                    self.heterozygous_segments = self.heterozygous_segments + np.sum(two_snps[position:][:, [1, 2, 3]], axis=0)
                    self.homozygous_segments = self.homozygous_segments + np.sum(two_snps[0:position][:, [1,2, 3]], axis=0)
                    break
            #Now to get the other arm
            if chr2[0] == 0:#If arm was 2L which had a BP extract het 2R
                self.heterozygous_segments = self.heterozygous_segments + np.sum(self.polarized_samples[sampleID][1][:,[1,2,3]], axis=0)
            else:#If arm was 2R which had BP extract a homozygous 2L
                self.homozygous_segments = self.homozygous_segments + np.sum(self.polarized_samples[sampleID][0][:, [1,2,3,]], axis=0)



        else:  # gamete was P2, P1
            for position in range(len(two_snps)):
                if two_snps[position][0] >= two_bp:
                    self.homozygous_segments = self.homozygous_segments + np.sum(two_snps[position:][:, [1, 2,3]], axis=0)
                    self.heterozygous_segments = self.heterozygous_segments + np.sum(two_snps[0:position][:, [1, 2,3]], axis=0)
                    break
            if chr2[0] == 0:#If arm was 2L which had a BP extract homozygous 2R
                self.homozygous_segments= self.homozygous_segments + np.sum(self.polarized_samples[sampleID][1][:,[1,2,3]], axis=0)
            else:#If arm was 2R which had BP extract a heterozygous 2L
                self.heterozygous_segments = self.heterozygous_segments + np.sum(self.polarized_samples[sampleID][0][:, [1,2,3,]], axis=0)

        ### Extract relevant SNP counts for chr3####
        #three_snps = self.polarized_samples[sampleID][chr3[0]]
        thre_bp = chr3[1]
        if chr3[2] == 1: #P1, P2

            for position in range(len(three_snps)):
                if three_snps[position][0] >= thre_bp:
                    self.heterozygous_segments = self.heterozygous_segments + np.sum(three_snps[position:][:, [1, 2, 3]], axis=0)
                    self.homozygous_segments = self.homozygous_segments + np.sum(three_snps[0:position][:, [1,2, 3]], axis=0)
                    break
            #Now to get the other arm
            if chr2[0] == 2:#If arm was 3L which had a BP extract het 3R
                self.heterozygous_segments = self.heterozygous_segments + np.sum(self.polarized_samples[sampleID][3][:,[1,2,3]], axis=0)
            else:#If arm was 3R which had BP extract a homozygous 3L
                self.homozygous_segments = self.homozygous_segments + np.sum(self.polarized_samples[sampleID][2][:, [1,2,3,]], axis=0)



        else:  # gamete was P2, P1
            for position in range(len(three_snps)):
                if three_snps[position][0] >= thre_bp:
                    self.homozygous_segments = self.homozygous_segments + np.sum(three_snps[position:][:, [1, 2,3]], axis=0)
                    self.heterozygous_segments = self.heterozygous_segments + np.sum(three_snps[0:position][:, [1, 2,3]], axis=0)
                    break
            if chr2[0] == 2:#If arm was 3L which had a BP extract homozygous 3R
                self.homozygous_segments= self.homozygous_segments + np.sum(self.polarized_samples[sampleID][3][:,[1,2,3]], axis=0)
            else:#If arm was 3R which had BP extract a heterozygous 3L
                self.heterozygous_segments = self.heterozygous_segments + np.sum(self.polarized_samples[sampleID][2][:, [1,2,3,]], axis=0)

        #All of the


    def load_SNP_array(self, snp_array, path, encoding='UTF-8'):
        snp_File = os.path.join(path, snp_array)
        SNPs = np.load(snp_File, encoding=encoding)

        return SNPs


    def load_reference(self, reference, path):
        np_file = os.path.join(path, reference)
        reference = np.load(np_file)

        return reference

    def polarizeSNPs(self, simSNP, refSNP, sampleID):

        """
        Construct an NP array for each of the single cells that converts or SNP calls into P1, P2, or error so we can
        simply analyze them.



        POS P1  P2  Err
        :param simSNP:
        :param refSNP:
        :return:
        """


        polarized_SNPs = []

        for chrom in range(5):
            assert len(simSNP[chrom][:, 0]) == len(refSNP[chrom][:, 0]), "Sample SNP array does not have same dimension as reference SNPs; Sample: {0}".format(sampleID)
            chrom_SNPs = np.zeros(shape=(len(simSNP[chrom][:,0]) , 4))
            chrom_SNPs[:,0] = simSNP[chrom][:,0]
            for snp in range(len(simSNP[chrom][:,0])):
                if simSNP[chrom][snp][1] == refSNP[chrom][snp][1]: #P1 allele
                    chrom_SNPs[snp][1] += 1
                elif simSNP[chrom][snp][1] == refSNP[chrom][snp][2]: #P2 allele
                    chrom_SNPs[snp][2] += 1
                else:# P1 allele
                    chrom_SNPs[snp][3] += 1
            polarized_SNPs.append(chrom_SNPs)

        polarized_SNPs = np.asarray(polarized_SNPs)

        return polarized_SNPs

    def readCO(self, tsv, path, sample_num=525):
        true_crossovers = np.zeros(shape=(sample_num, 3))
        add_Mb = {'2L': 0, '2R': 23000000, '3L': 0, '3R': 24500000, 'X': 0}
        full_tsv = os.path.join(path, tsv)
        with open(full_tsv, 'r') as myTsv:
            TSV_reader = csv.reader(myTsv, delimiter='\t')
            next(TSV_reader)
            index = 0
            for field in TSV_reader:

                chr2 = (int(field[2]) + add_Mb[field[1]]) /1000000
                chr3 = (int(field[5]) + add_Mb[field[4]]) / 1000000
                chrX = int(field[8]) / 1000000

                true_crossovers[index] = np.asarray([chr2, chr3, chrX])
                index += 1
        return true_crossovers


    def subSample_SNP(self, snp_input, sampling):
        all_subsamples = []
        for chrom in range(5):
            #sample random indices from our SNP array
            rand_snps = sorted(random.sample(range(len(snp_input[chrom][:,0])-1), int(sampling*len(snp_input[chrom][:,0]))))

            #extract SNP features:
            sub_sampledSNPS = np.zeros(shape=(len(rand_snps), 4))

            for snp in range(len(rand_snps)):
                sub_sampledSNPS[snp] = snp_input[chrom][rand_snps[snp]]
            all_subsamples.append(sub_sampledSNPS)

        all_subsamples = np.asarray(all_subsamples)


        return all_subsamples


    def cluster_RBPs(self, predictions, truth, all_pointers=False, chr2_dist=1000, chr3_dist=1000, chrX_dist=1000):
        """
        An algorithm for clustering our 3 dimensional points in space.

        Clusters will be called by a maximum distance cut off predetermined by the input parameters of this function.
        I have designed the function to have 3 seperate minimum distance parameters for each chromosome, but the way that
        the distance in 3d space is formulated I don't think it will make that much of a distance.

        Once you have the distances computed then we must simply aglomerate all of the individuals that are called from the same breakpoints


        :return:
        """

        #Compute the mean 3d distance from our min dist for each bp
        distance = math.sqrt(chr2_dist**2 + chr3_dist**2 + chrX_dist**2) / 1000000

        #Cluster by a min distance between points:
        cluster_pred = fclusterdata(predictions, distance , criterion='distance') #This works well we must determine what is a reasonable minimum euclidean distance between points however



        #### Get all cluster predictions and cells into a table:
        all_clusters = {}

        for cell in range(len(cluster_pred)):
            if cluster_pred[cell] not in all_clusters.keys():
                all_clusters[cluster_pred[cell]] = [cell]
            else:
                all_clusters[cluster_pred[cell]].append(cell)

        #Clusters and sample IDs are tabulated now we can proceed on confirming their identity

        #For each cluster in order to confirm the correct identity we should use the state space of the Viterbi decoding to determine whether or not the segments are P1, P2 or P2, P1
        #If clusters have similar breakpoints but different ordering then they are distinct individuals


        for clust in all_clusters.keys():
            if len(all_clusters[clust]) > 1:
                #Check viterbi of each:
                for cell in all_clusters[clust]:
                    for chromosome in range(3):
                        pass






        #Print out clusters
        #for i in duplicate_cells:
        #    print('{0}\t{1}'.format(i[0], i[1]))

def polarize_SNP_array(ref, snp):


    ref_path, ref_file = os.path.split(ref)

    snp_path, snp_file = os.path.split(snp)
    myAnalysis = analyzeSEQ()

    reference_genome = myAnalysis.load_reference(path=ref_path, reference=ref_file)


    SNP_data = myAnalysis.load_SNP_array(snp_array=snp_file, path=snp_path, encoding='latin1')

    pol_analysis = []
    ID = 1
    for simul in SNP_data:
        snp_array = myAnalysis.polarizeSNPs(refSNP=reference_genome, simSNP=simul, sampleID=ID)
        pol_analysis.append(snp_array)
        ID +=1



    #Save my simulated data
    save_samples = np.asarray(pol_analysis)
    data_path = os.path.join(snp_path, 'POLARIZED_' + snp_file)
    np.save(data_path, save_samples)



def train():
    #Use every other simulated data set to train

    myAnalysis = analyzeSEQ()
    myAnalysis.polarized_samples = myAnalysis.load_SNP_array(path='/home/iskander/Documents/MEIOTIC_DRIVE/', snp_array='polarized_simulations.npy', encoding='latin1')
    pi_P1, pi_P2 = myAnalysis.supervisedTraining(path="/home/iskander/Documents/MEIOTIC_DRIVE/",
                                  crossovers="DGRP_882_129_simulations_3_19_19_crossovers.tsv")

    print(pi_P1)
    print(pi_P2)
def detect_simulated(snp):

    snp_path, snp_file = os.path.split(snp)
    myAnalysis = analyzeSEQ()
    myAnalysis.polarized_samples = myAnalysis.load_SNP_array(path=snp_path, snp_array=snp_file, encoding='latin1')
    #true_CO = myAnalysis.readCO(path="/home/iskander/Documents/MEIOTIC_DRIVE/", tsv="DGRP_882_129_simulations_3_19_19_crossovers.tsv")

    ###### Instaniate arrays #####
    cell_predictions = np.zeros(shape=(len(myAnalysis.polarized_samples), 3))
    cell_prediction_indices = np.zeros(shape=(len(myAnalysis.polarized_samples), 3))

    ###### Perform analysises on sub sampled SNP arrays ####
    #Simulate coverage in 20,000 - 50,000 unique reads/cell
    #
    cov_values = np.random.exponential(.015, len(myAnalysis.polarized_samples))
    avg_cov = np.average(cov_values)
    filtered_cells = [cov for cov in cov_values if cov > .008]
    
    #Predict BPs
    index = 0
    for cell in myAnalysis.polarized_samples:
        sub_sampling = myAnalysis.subSample_SNP(cell, sampling=cov_values[index])
        cell_pointers = myAnalysis.hmmViterbi(snp_input=sub_sampling)
        ML_predicts, ML_indices = myAnalysis.decode(cell_pointers, sub_sampling)
        cell_predictions[index] = np.asarray(ML_predicts)
        cell_prediction_indices[index] = np.asarray(ML_indices)
        index += 1

    #Cluster the cells based on RBPs and a minimum distance
    myAnalysis.cluster_RBPs()



def hmm_Testing():
    #Going to design a function test my viterbi decoding at an exhaustive regime of coverage
    myAnalysis = analyzeSEQ()
    myAnalysis.polarized_samples = myAnalysis.load_SNP_array(path='/home/iskander/Documents/MEIOTIC_DRIVE/',
                                                             snp_array='polarized_simulations.npy', encoding='latin1')
    true_CO = myAnalysis.readCO(path="/home/iskander/Documents/MEIOTIC_DRIVE/",
                                tsv="DGRP_882_129_simulations_3_19_19_crossovers.tsv")

    all_cellArray = []
    predict_array = np.zeros(shape=(100, 3))

    cell_index = 0
    index = 0
    cell = myAnalysis.polarized_samples[0]
    for test in range(1,101):


        cov = .05/100 * test
        sub_sampling = myAnalysis.subSample_SNP(cell, sampling=cov)
        cell_pointers = myAnalysis.hmmViterbi(snp_input=sub_sampling)
        ML_predicts, ML_indices = myAnalysis.decode(cell_pointers, sub_sampling)
        predict_array[index] = ML_predicts

        index += 1
    cell_index += 1

    np.save(os.path.join('/home/iskander/Documents/MEIOTIC_DRIVE/','coverage_regimes.npy'), predict_array)


def plotting():

    myAnalysis = analyzeSEQ()
    true_CO = myAnalysis.readCO(path="/home/iskander/Documents/MEIOTIC_DRIVE/",
                                tsv="DGRP_882_129_simulations_3_19_19_crossovers.tsv")


    ml_array = np.load(os.path.join('/home/iskander/Documents/MEIOTIC_DRIVE','coverage_regimes.npy'))
    ml2_array = np.load(os.path.join('/home/iskander/Documents/MEIOTIC_DRIVE', 'coverage_samples.npy'))

    cov_array = [.05/100*cov for cov in range(1, 101)]

    all_dist = np.zeros(shape=(100, 4))
    index = 0
    for ML_predicts in ml_array:

        distance = math.sqrt((true_CO[0][0] - ML_predicts[0]) ** 2 + (true_CO[0][1] - ML_predicts[1]) ** 2 + (true_CO[0][2] - ML_predicts[2]) ** 2)  # calculate distance between estimates

        x_dist = abs(true_CO[0][2] - ML_predicts[2])
        chr2_dist = abs(true_CO[0][0] - ML_predicts[0])
        chr3_dist = abs(true_CO[0][1] - ML_predicts[1])
        distances = np.asarray([distance, chr2_dist, chr3_dist, x_dist])


        all_dist[index] = distances
        index += 1

    distance = math.sqrt((true_CO[0][0] - ml2_array[4][0]) ** 2 + (true_CO[0][1] - ml2_array[4][1]) ** 2 + (
            true_CO[0][2] - ml2_array[4][2]) ** 2)  # calculate distance between estimates

    x_dist = abs(true_CO[0][2] - ml2_array[4][2])
    chr2_dist = abs(true_CO[0][0] - ml2_array[4][0])
    chr3_dist = abs(true_CO[0][1] - ml2_array[4][1])

    print(all_dist[cov_array.index(.01)])
    print(np.asarray([distance, chr2_dist, chr3_dist, x_dist]))

    #######################


    titles = ['3d distance', 'Chr2 distance', 'Chr3 distance', 'ChrX distance']
    #Lineplot distance vs. coverage
    with sns.axes_style('darkgrid'):
        fig = plt.figure(1, figsize=(10,5))
        axs = fig.subplots(4,1, sharex=True, sharey=True)
    for d in range(4):
        sns.lineplot(cov_array, all_dist[:,d], ax = axs[d])
        axs[d].set_title('{0}'.format(titles[d]))
        axs[d].set_ylabel('Distance (Mb)')
        #axs[d].set_xlim(0, .06)
    plt.xlabel('Coverage')

    plt.suptitle('Distance from True RBP vs. Coverage')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join('/home/iskander/Documents/MEIOTIC_DRIVE','TEST_dist_cov.png'), dpi=200)

    #3d scatter plot
    fig3d = plt.figure(2, figsize=(20,20))
    ax = fig3d.add_subplot(111, projection='3d')
    ax.scatter(ml_array[:,0], ml_array[:,1], ml_array[:,2], marker='o')
    ax.scatter(true_CO[0][0], true_CO[0][1], true_CO[0][2], c='red', marker='o', s=50)
    plt.savefig(os.path.join('/home/iskander/Documents/MEIOTIC_DRIVE','3d_cov.png'), dpi=200)

    #Line plot but on the same axes
    with sns.axes_style('darkgrid'):
        fig3 = plt.figure(3, figsize=(10,5))
        #axs = fig.subplots(4,1, sharex=True, sharey=True)
    for d in range(4):
        sns.lineplot(cov_array, all_dist[:,d])
    plt.title('Distance v. Coverage')
    plt.xlabel('Coverage')
    plt.ylabel('Distance (Mb)')
    plt.ylim((0,1.2))
    plt.figlegend(titles)
    plt.savefig(os.path.join('/home/iskander/Documents/MEIOTIC_DRIVE','test_cov_dist_plot2.png'), dpi=200)


if __name__ == '__main__':

    myArgs = CommandLine()
    if myArgs.args.polarize == True:#If program is called to polarize the SNPs in preprocessing
        polarize_SNP_array(ref=myArgs.args.reference, snp=myArgs.args.snp)
    else:
        detect_simulated(snp=myArgs.args.snp)
start = time.time()

end = time.time() - start
print(end)
