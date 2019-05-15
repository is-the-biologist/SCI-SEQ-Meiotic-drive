import os
import numpy as np
import csv
import operator
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import scipy.stats
import time
import sys
from scipy.cluster.hierarchy import fclusterdata
import argparse
from multiprocessing import Pool

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
                                 default='/home/iskander/Documents/MEIOTIC_DRIVE/polarized_simulations.npy')
        self.parser.add_argument("-r", "--reference", type=str, action="store", nargs="?",
                                 help="The reference genome numpy file to polarize SNPs",
                                 default='/home/iskander/Documents/MEIOTIC_DRIVE/882_129.snp_reference.npy')
        self.parser.add_argument("-t", "--threads", type=int, action="store", nargs="?",
                                 help="Number of threads to be specified if multi-threading is to be used.",
                                 default=0)
        if inOpts is None:  ## DONT REMOVE THIS. I DONT KNOW WHAT IT DOES BUT IT BREAKS IT
            self.args = self.parser.parse_args()
        else:
            self.args = self.parser.parse_args(inOpts)




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
        #P1/P2 heterozygous
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
        sys.stdout.write('Decoding maximum likelihood path...\n')
        rbp_positions= np.zeros(shape=(1,3)) #chr2, chr3, chrx
        rbp_switches= np.zeros(shape=(1,3))
        rbp_indices = np.zeros(shape=(1, 3))

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
                    breakpoint = all_chr_arms[chr_index][position-1] + abs(all_chr_arms[chr_index][position] - all_chr_arms[chr_index][position-1])/2 #take midpoint from the switch over position
                    rbp_positions[0][chr_index] = breakpoint
                    rbp_indices[0][chr_index] = position
                    rbp_switches[0][chr_index] = traceback[position]

            chr_index += 1
        sys.stdout.write('Recombination breakpoints predicted!\n')
        return rbp_positions, rbp_switches, rbp_indices
    def hmmFwBw(self, snp_input):

        sys.stdout.write('Calculating posterior probability of HMM...\n')
        states = ['p1', 'p2']
        chr_posteriors = []

        chr_2 = np.concatenate((snp_input[0][:, 0], snp_input[1][:, 0] + 23000000))
        chr_3 = np.concatenate((snp_input[2][:, 0], snp_input[3][:, 0] + 24500000))
        chr_x = snp_input[4][:, 0]
        all_chr_arms = [chr_2, False, chr_3, False,
                        chr_x]  # Have to put these placeholders in this because I have developed a weird codebase and I reap what I so


        # Fwd Bckwd
        sys.stdout.write('Computing forward-backward algorithm...')
        for chrom in range(0, 5, 2):  # consider each chromosome as a single sequence
            if chrom != 4:
                chrom_length = len(snp_input[chrom]) + len(snp_input[chrom + 1])
            else:
                chrom_length = len(snp_input[chrom])

            fMatrix = np.zeros(shape=(2, chrom_length))


            # First state
            # P(state) * P(emission|state)
            init_emission = int(snp_input[chrom][0][1])

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


                emission = int(chromosome_arm[SNP][1])
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
                    emission = int(chromosome_arm[SNP+1][1])

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


        sys.stdout.write('Posterior probability matrix constructed.\n')
        return chr_posteriors

    def draw(self, posterior_matrix, snp_matrix, predicted_rbps, truth, interval, title='TestID'):
        chr_decoding = {0:'Chr2', 1:'Chr3', 2:'ChrX'}

        chr_2 = np.concatenate((snp_matrix[0][:, 0], snp_matrix[1][:, 0] + 23000000)) / 1000000
        chr_3 = np.concatenate((snp_matrix[2][:, 0], snp_matrix[3][:, 0] + 24500000)) / 1000000
        chr_x = snp_matrix[4][:, 0] / 1000000
        all_chr_arms = [chr_2, chr_3, chr_x]


        chr2_alleles = np.concatenate((snp_matrix[0], snp_matrix[1]))[:,1]
        chr3_alleles = np.concatenate((snp_matrix[2], snp_matrix[3]))[:,1]
        chrX_alleles = snp_matrix[4][:,1]

        all_alleles = [chr2_alleles, chr3_alleles, chrX_alleles]
        with sns.axes_style('darkgrid'):
            fig = plt.figure(figsize=(20, 10))
            axs = fig.subplots(3,2)


        for arm in range(3):
            left_interval = all_chr_arms[arm][int(interval[arm][0])]
            right_interval = all_chr_arms[arm][int(interval[arm][1])]

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

            axs[arm][0].axvline(left_interval, linestyle='--', color='green')
            axs[arm][0].axvline(right_interval, linestyle='--', color='green')

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

        sys.stdout.write('Computing Viterbi decoding through HMM state space...')

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

            init_emission = int(snp_input[chrom][0][1])

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
                emission = int(chromosome_arm[SNP][1])
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
        SNPs = np.load(snp_File, encoding=encoding, allow_pickle=True)
        sys.stdout.write('SNP input loaded...\n')
        return SNPs


    def load_reference(self, reference, path):
        np_file = os.path.join(path, reference)
        reference = np.load(np_file,  allow_pickle=True)

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
            chrom_SNPs = np.zeros(shape=(len(simSNP[chrom][:,0]) , 2))
            chrom_SNPs[:,0] = simSNP[chrom][:,0]
            for snp in range(len(simSNP[chrom][:,0])):
                if simSNP[chrom][snp][1] == refSNP[chrom][snp][1]: #P1 allele
                    chrom_SNPs[snp][1] = 0
                elif simSNP[chrom][snp][1] == refSNP[chrom][snp][2]: #P2 allele
                    chrom_SNPs[snp][1] = 1
                else:# error allele
                    chrom_SNPs[snp][1] = 2
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


    def calc_interval(self, posterior_prob, ML_ind, ML_switch, threshold=.99):
        """
        Use the inherent properties of the posterior probabilities to draw some intervals around our predicted Maximum likelihood detected viterbi interval.

        To determine the range of values where our breakpoint could be we take our

        :param posterior_prob:
        :param ML:
        :return:
        """


        sys.stdout.write('Calculating intervals around recombination breakpoint with posterior probability threshold of {0}.\n'.format(threshold))
        intervals = np.zeros(shape=(3,2))

        for chromosome in range(3):
            prediction = int(ML_ind[0][chromosome])
            switch = int(ML_switch[0][chromosome])
            interval = -10
            if switch == 0:
                alt_switch = 1
            else:
                alt_switch = 0

            ##### Leftmost interval end point
            while interval < np.log(threshold): #Iterate through the left of the viterbi state change until we reach a .99 probability of the alternative state
                interval = posterior_prob[chromosome][:,prediction][alt_switch]
                prediction = prediction - 1
                if prediction <= 0: #if we reach the end of the SNPs w/o reaching our condition break at the end
                    prediction = 0
                    break
            interval_endpoint_left = prediction

            ##### Rightmost interval end point
            interval = -10
            prediction = int(ML_ind[0][chromosome]) #Iterate through the right side of the viterbi state change until we reach a .99 probability of the changed state
            while interval < np.log(threshold):
                interval = posterior_prob[chromosome][:, prediction][switch]
                prediction = prediction + 1
                if prediction >= len(posterior_prob[chromosome][0,:]):
                    prediction = len(posterior_prob[chromosome][0,:]) - 1
                    break
            interval_endpoint_right = prediction

            intervals[chromosome] = np.asarray([interval_endpoint_left, interval_endpoint_right])


        return intervals

    def cluster_RBPs(self, predictions, snp_pos, distance=.1):
        """
        An algorithm for clustering our 3 dimensional points in space.

        Clusters will be called by a maximum distance cut off predetermined by the input parameters of this function.
        Distance parameter is in Mb and may be dependent on the SNP density of the single cell sequences.
        Once you have the distances computed then we must simply aglomerate all of the individuals that are called from the same breakpoints


        :return:
        """


        sys.stdout.write('Clustering individuals by {0} Mb distance...\n'.format(distance))
        #Cluster by a min distance between points:
        cluster_pred = fclusterdata(predictions, distance , criterion='distance')



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
        sys.stdout.write('Found initial {0} clusters.\n Checking for errors...\n'.format(len(all_clusters.keys())))
        de_cluster = {}
        for clust in all_clusters.keys():
            if len(all_clusters[clust]) > 1:
                #Check viterbi switching of each:
                switches = [tuple(snp_pos[ID]) for ID in all_clusters[clust]]

                if len(set(switches)) == 1:#If all points in the cluster also have the same switch from P1, P2 or vice versa then we are likely calling our clusters correctly.
                    pass
                else: #Decluster the incorrect clusters

                    #There may be a subset of individuals of this cluster that clustered correctly so decluster based on the switches
                    sub_clusters = {switch_key:[] for switch_key in switches}
                    for ID in all_clusters[clust]:
                        sub_clusters[tuple(snp_pos[ID])].append(ID)

                    #Now the we have determined the subsets of individuals in these sub clusters we will create a new cluster for each subcluster
                    sorted_subCluster_keys = sorted(list(sub_clusters.keys()))#order dict.keys() so maintain ordering across iteration
                    de_cluster[clust] = sub_clusters[sorted_subCluster_keys[0]] #sub_cluster 1 is arbitrarily the original cluster
                    for sc in range(1, len(sub_clusters.keys())):
                        clust_id = len(all_clusters.keys()) + sc
                        de_cluster[clust_id] = sub_clusters[sorted_subCluster_keys[sc]]
        #Iterated through all of the clusters that must be de_clustered and stored them in another table now we will fix our all_clusters table
        for dc in de_cluster.keys():
            all_clusters[dc] = de_cluster[dc]

        sys.stdout.write('After checking errors found a total of {0} clusters...\n'.format(len(all_clusters.keys())))
        #Clusters are now called and corrected for easy errors now lets return this tabular information
        return all_clusters

    def avgInterval(self, snp_matrix, interval):
        """
        Calculate the average size of our posterior prob intervals for all RBP predictions.

        :param snp_matrix:
        :param interval:
        :return:
        """

        chr_2 = np.concatenate((snp_matrix[0][:, 0], snp_matrix[1][:, 0] + 23000000)) / 1000000
        chr_3 = np.concatenate((snp_matrix[2][:, 0], snp_matrix[3][:, 0] + 24500000)) / 1000000
        chr_x = snp_matrix[4][:, 0] / 1000000
        all_chr_arms = [chr_2, chr_3, chr_x]

        genome_distance = 0
        for arm in range(3):
            left_interval = all_chr_arms[arm][int(interval[arm][0])]
            right_interval = all_chr_arms[arm][int(interval[arm][1])]
            genome_distance += abs(left_interval - right_interval)

        avg_distance = genome_distance/3

        return avg_distance

    def filterNaN(self, snp_input):
        """
        This method cleans up the SNPs that are missing from our input and outputs only the present data structures.

        :param snp_input:
        :return:
        """
        chroms = ['2L', '2R', '3L', '3R', 'X']
        sys.stdout.write('Filtering missing data...\n')
        clean_Cell = []
        total_markers = 0
        total_missing = 0
        for chrom in range(5):
            #Determine where the missing data is:
            missing_data = np.argwhere(np.isnan(snp_input[chrom][:,1]))
            clean_chrom = np.delete(snp_input[chrom], missing_data, 0)
            clean_Cell.append(clean_chrom)
            total_markers += len(snp_input[chrom][:,1])
            total_missing += len(missing_data)
            #sys.stdout.write('For chromosome {2}: {0} markers missing out of {1} total..\n'.format(len(missing_data), len(snp_input[chrom][:,1]), chroms[chrom]))
        clean_Cell = np.asarray(clean_Cell)

        sys.stdout.write('{0} markers present out of {1} total...\n'.format(total_markers - total_missing, total_markers))
        return clean_Cell

    def mergeIndividuals(self, clusters):
        """
        After clusters are called we will merge the SNP inputs of the individuals that had clusters with more than one individual.
        This will fill out some of the missing data and will increase our certainty on the breakpoint predictions when we re-run our HMM and see if we get a change in clustering.


        :return:
        """

        orig_indivs = len(self.polarized_samples)
        sys.stdout.write('Initializing cluster merge...\n')
        merged_clusters = []
        for mc in clusters.keys():
            if len(clusters[mc]) > 1:

                str_inds = [str(indiv+1) for indiv in clusters[mc]]
                sysout = ', '.join(str_inds)
                sys.stdout.write('Merging individuals {0}\n'.format(sysout))
                merged_genome = []

                for arm in range(5):
                    merged_arm = self.polarized_samples[clusters[mc][0]][arm]
                    for individual in range(1, len(clusters[mc])):
                        #Merge the SNP calls from each chromosome arm of each individual
                        merged_arm = np.append(merged_arm, self.polarized_samples[clusters[mc][individual]][arm][:,1][..., None], 1)

                    #Merged all the SNP calls now we will check the consensus of each allele
                    arm_calls = np.zeros(shape=(len(merged_arm[:,0]), 2))
                    arm_calls[:,0] = merged_arm[:,0]

                    snp_index = 0
                    for SNP in merged_arm[:,1:]:
                        homozygous_Model = [self.pi_p1_1, self.pi_p1_2, self.pi_p1_3]
                        heterozygous_Model = [self.pi_p2_1, self.pi_p2_2, self.pi_p2_3]

                        #We will construct a likelihood ratio test between or homozygous and heterozygous model to determine the likelihood of the position given the allele frequency
                        L_homo = 0
                        L_het = 0
                        for call in SNP:
                            #sum log probs
                            if np.isnan(call) == False:#filter missing data
                                L_homo += np.log(homozygous_Model[int(call)])
                                L_het += np.log(heterozygous_Model[int(call)])
                            else:
                                pass
                        #Likelihoods for each model are calculated and now can attempt a test
                        likelihood = -2*(L_homo - L_het)
                        #Now perform a likelihood ratio and do a hard cutoff at 0 ... this may require a bit more fine tuning later on
                        if likelihood == 0:#Check for NaN arrays and arrays with genotype errors
                            if np.isnan(SNP).all() == True:
                                merged_call = np.nan
                            else:
                                merged_call = 2
                        elif likelihood > 0:#call the site as heterozygous and assign a P2 allele to it
                            merged_call = 1

                        elif likelihood < 0:#call the site as homozygous and assign a P1 allele to it
                            merged_call = 0
                        arm_calls[:,1][snp_index] = merged_call
                        snp_index += 1
                    merged_genome.append(arm_calls)
                merged_genome = np.asarray(merged_genome)
                merged_clusters.append(merged_genome)
        merged_clusters = np.asarray(merged_clusters)
        #We have finished constructing our new merged genomes now we will delete our old np arrays from our matrix and add our new genomes to it
        delete_elements = [individual for c in clusters.keys() for individual in clusters[c] if len(clusters[c]) > 1]

        self.polarized_samples = np.delete(self.polarized_samples, delete_elements, 0)
        self.polarized_samples = np.append(self.polarized_samples, merged_clusters, 0)
        new_indivs = len(self.polarized_samples)
        sys.stdout.write('Merged {0} individuals into {1} individuals...\n'.format(orig_indivs, new_indivs))

def polarize_SNP_array(ref, snp):


    ref_path, ref_file = os.path.split(ref)

    snp_path, snp_file = os.path.split(snp)
    myAnalysis = analyzeSEQ()

    reference_genome = myAnalysis.load_reference(path=ref_path, reference=ref_file)
    sys.stdout.write('Reference loaded...\n')

    SNP_data = myAnalysis.load_SNP_array(snp_array=snp_file, path=snp_path, encoding='latin1')


    pol_analysis = []
    ID = 1
    sys.stdout.write('Preparing to polarize...\n{0} individuals to polarize\n'.format(len(SNP_data)))
    for simul in SNP_data:
        snp_array = myAnalysis.polarizeSNPs(refSNP=reference_genome, simSNP=simul, sampleID=ID)
        pol_analysis.append(snp_array)
        sys.stdout.write('{0} individuals out of {1} polarized\n'.format(ID, len(SNP_data)))
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


def singleThread():


    #true_CO = myAnalysis.readCO(path="/home/iskander/Documents/Barbash_lab/mDrive/", tsv="out_crossovers.tsv")

    ###### Instaniate arrays #####
    cell_predictions = np.zeros(shape=(len(myAnalysis.polarized_samples), 3))
    cell_prediction_switches = np.zeros(shape=(len(myAnalysis.polarized_samples), 3))

    ###### Perform analysises on sub sampled SNP arrays ####
    #Simulate coverage in 20,000 - 50,000 unique reads/cell
    #Given that this cross has a SNP ~200 bp apart and a read spans 75bp we should sample like 1 snp every ~2.5 reads
    #So we should see 8,000 - 20,000 SNPs per genome


    #Predict BPs
    sys.stdout.write('Initializing HMM to detect recombination breakpoints...\n')
    index = 0
    sample_distAvg = 0
    for cell in myAnalysis.polarized_samples:

        sub_sampling = myAnalysis.filterNaN(cell)
        cell_pointers = myAnalysis.hmmViterbi(snp_input=sub_sampling)
        ML_predicts, ML_switches, ML_indices = myAnalysis.decode(cell_pointers, sub_sampling)
        cell_predictions[index] = np.asarray(ML_predicts)
        cell_prediction_switches[index] = np.asarray(ML_switches)
        posterior = myAnalysis.hmmFwBw(sub_sampling)
        ML_intervals = myAnalysis.calc_interval(posterior, ML_indices, ML_switches, threshold=.99)
        sample_distAvg += myAnalysis.avgInterval(sub_sampling, ML_intervals)
        #myAnalysis.draw(posterior, sub_sampling, cell_predictions[index], true_CO[index], ML_intervals, title='ID:{0}_test_markers'.format(index+1))
        sys.stdout.write('Breakpoint predicted for {0} out of {1} individuals.\n'.format(index+1, len(myAnalysis.polarized_samples)))
        index += 1
    sample_distAvg = sample_distAvg/index
    sys.stdout.write('HMM complete...\n Recombination breakpoints detected.\nNow initializing clustering of individuals...\n')
    #Cluster the cells based on RBPs and a minimum distance
    clusters = myAnalysis.cluster_RBPs(predictions=cell_predictions, snp_pos=cell_prediction_switches, distance=sample_distAvg)
    myAnalysis.mergeIndividuals(clusters)


def hmm_Testing():
    #Going to design a function test my viterbi decoding at an exhaustive regime of coverage
    myAnalysis = analyzeSEQ()
    myAnalysis.polarized_samples = myAnalysis.load_SNP_array(path='/home/iskander/Documents/MEIOTIC_DRIVE/',
                                                             snp_array='POLARIZED_out.npy', encoding='latin1')


    all_cellArray = []
    predict_array = np.zeros(shape=(100, 3))
    interval_array = np.zeros(shape=(100, 6))
    cell_index = 0
    index = 0
    cell = myAnalysis.polarized_samples[0]
    for test in range(100):


        cov = 8000 - 80*test
        sub_sampling = myAnalysis.subSample_SNP(cell, sampling=cov)
        cell_pointers = myAnalysis.hmmViterbi(snp_input=sub_sampling)
        ML_predicts, ML_indices, ML_switches = myAnalysis.decode(cell_pointers, sub_sampling)
        predict_array[index] = ML_predicts
        posterior = myAnalysis.hmmFwBw(sub_sampling)
        ML_intervals = myAnalysis.calc_interval(posterior, ML_indices, ML_switches, threshold=.99)

        index += 1
    cell_index += 1

    np.save(os.path.join('/home/iskander/Documents/MEIOTIC_DRIVE/','coverage_regimes_intervals.npy'), predict_array)



def multiThreaded(cell_snp):

    sub_sampling = myAnalysis.filterNaN(snp_input=cell_snp)
    cell_pointers = myAnalysis.hmmViterbi(snp_input=sub_sampling)
    ML_predicts, ML_switches, ML_indices = myAnalysis.decode(cell_pointers, sub_sampling)
    posterior_probs = myAnalysis.hmmFwBw(sub_sampling)
    ML_intervals = myAnalysis.calc_interval(posterior_probs, ML_indices, ML_switches)
    dist_avg = myAnalysis.avgInterval(sub_sampling, ML_intervals)

    return ML_predicts, ML_switches, dist_avg



if __name__ == '__main__':
    start = time.time()
    myArgs = CommandLine()
    if myArgs.args.polarize == True:#If program is called to polarize the SNPs in preprocessing
        sys.stdout.write('Polarizing SNPs...\nReference:{0}\nSNP input:{1}\n'.format(myArgs.args.reference, myArgs.args.snp))

        polarize_SNP_array(ref=myArgs.args.reference, snp=myArgs.args.snp)
    else:
        sys.stdout.write('Initializing single cell analysis...\nSNP input:{0}\n'.format(myArgs.args.snp))
        if myArgs.args.threads == 0:
            sys.stdout.write('Single thread option chosen.\n')
            snp_path, snp_file = os.path.split(myArgs.args.snp)
            myAnalysis = analyzeSEQ()
            myAnalysis.polarized_samples = myAnalysis.load_SNP_array(path=snp_path, snp_array=snp_file, encoding='latin1')

            #### Single thread ######
            singleThread()
        else:
            #### Multithreaded ####
            sys.stdout.write('Multithread option chosen with {0} threads.\n'.format(myArgs.args.threads))
            snp_path, snp_file = os.path.split(myArgs.args.snp)
            myAnalysis = analyzeSEQ()
            myAnalysis.polarized_samples = myAnalysis.load_SNP_array(path=snp_path, snp_array=snp_file, encoding='latin1')
            cell_predictions = np.zeros(shape=(len(myAnalysis.polarized_samples), 3))
            cell_prediction_switches = np.zeros(shape=(len(myAnalysis.polarized_samples),3))

            #Run the functions
            with Pool(processes=myArgs.args.threads) as myPool:
                ML_result = myPool.map(multiThreaded, myAnalysis.polarized_samples)

            sample_distAvg = 0
            for pred in range(len(ML_result)):

                cell_predictions[pred] = ML_result[pred][0]
                cell_prediction_switches[pred] = ML_result[pred][1]
                sample_distAvg += ML_result[pred][2]
            sample_distAvg = sample_distAvg / len(ML_result)
            #Cluster method:
            clusters = myAnalysis.cluster_RBPs(predictions=cell_predictions, snp_pos=cell_prediction_switches, distance=sample_distAvg)
            myAnalysis.mergeIndividuals(clusters)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    sys.stdout.write("Time elapsed: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
