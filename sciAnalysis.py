import os
import numpy as np
import csv
import operator
import math
import matplotlib.pyplot as plt
import seaborn as sns



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
        self.pi_p1_1 = 0.999304824583 #P1 allele in P1 homozygoous region
        self.pi_p1_2 = 0.000226259923442 #P2 allele in P1 homozygous region
        self.pi_p1_3 = 0.000468915493558 #Seq error in P1 homozygous region

        self.pi_p2_1 = 0.500478305194 #P1 allele in P1/P2
        self.pi_p2_2 = 0.49902244194 #P2 allele in P1/P2
        self.pi_p2_3 = 0.000499252866 #Seq error in P1/P2
        self.mu = .99
        ###Probability states for the HMM###

        self.init_chrom_states = {'p1':np.log(0.5), 'p2':np.log(0.5)}
        self.state_probs = {'p1':[np.log(self.pi_p1_1), np.log(self.pi_p1_2), np.log(self.pi_p1_3)],
                       'p2': [np.log(self.pi_p2_1), np.log(self.pi_p2_2), np.log(self.pi_p2_3)]
                       }
        self.transitions_probs ={'p1':{'p1':np.log(self.mu), 'p2':np.log(1-self.mu)}, 'p2':{'p1': np.log(1-self.mu), 'p2':np.log(self.mu)}}

    def decode(self, pointers, fileName, path):

        for chromosome in range(0, 5, 2):
            pass
    def hmmFwBw(self, snp_input):
        states = ['p1', 'p2']
        chr_posteriors = []

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
    def draw(self, posterior_matrix, snp_matrix):

        chr_2 = np.concatenate((snp_matrix[0][:,0],snp_matrix[1][:,0] + 23000000)) / 1000000
        fig = plt.figure(figsize=(15, 5))
        #sns.lineplot(chr_2, posterior_matrix[0][0])
        sns.lineplot(chr_2, posterior_matrix[0][1])
        plt.title('Posterior Probabilities of P1/P2 (Chr 2)')
        plt.ylabel('-log(probability)')
        plt.xlabel('Position (Mb)')
        plt.axvline(16.319859, linestyle='--', color='red')
        legend = ['P1/P2 Probability', 'Recombination breakpoint (16,319,859 bp)']
        plt.xlim((10, 20))
        plt.figlegend(legend)

        plot_name = os.path.join('/home/iskander/Documents/MEIOTIC_DRIVE/', 'test2_posteriors.png')
        plt.savefig(plot_name, dpi=200)

    def hmmViterbi(self, snp_input):

        """
        This function will perform our viterbi decoding for our recombination breakpoint detection. Depending on how fuzzy the breakpoint positions end up looking
        it may just be better to use the posterior distribution of the states and take like an area around the inflection point of the changes in posterior probabilities.


        :param snp_input:
        :return:
        """

        states = ['p1', 'p2']
        arm_pointers = []

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

                    self.extractSegments(chrIndex_2, chrIndex_3, chrIndex_X, sampleIndex)

                else:
                    pass
                skip_count += 1

            myCO.close()


        pi_P1 = self.homozygous_segments / np.sum(self.homozygous_segments)
        pi_P2 = self.heterozygous_segments / np.sum(self.heterozygous_segments)

        return pi_P1, pi_P2
    def extractSegments(self, chr2, chr3, chrX, sampleID):


        ###Extract relevant SNP counts for ChrX####

        x_snps = self.polarized_samples[sampleID][chrX[0]]
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
        two_snps = self.polarized_samples[sampleID][chr2[0]]
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
        three_snps = self.polarized_samples[sampleID][chr3[0]]
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



def polarize_SNP_array(path='/home/iskander/Documents/MEIOTIC_DRIVE/'):

    myAnalysis = analyzeSEQ()

    reference_genome = myAnalysis.load_reference(path=path, reference='882_129.snp_reference.npy')


    SNP_data = myAnalysis.load_SNP_array(snp_array='DGRP_882_129_simulations_3_19_19.npy', path=path, encoding='latin1')

    pol_analysis = []
    ID = 1
    for simul in SNP_data:
        snp_array = myAnalysis.polarizeSNPs(refSNP=reference_genome, simSNP=simul, sampleID=ID)
        pol_analysis.append(snp_array)
        ID +=1



    #Save my simulated data
    save_samples = np.asarray(pol_analysis)
    data_path = os.path.join(path, 'polarized_simulations.npy')
    np.save(data_path, save_samples)



def train():
    #Use every other simulated data set to
    data_path = os.path.join('/home/iskander/Documents/MEIOTIC_DRIVE/', 'polarized_simulations.npy')
    sim_data = np.load(data_path)
    myAnalysis = analyzeSEQ()
    myAnalysis.polarized_samples = sim_data
    pi_P1, pi_P2 = myAnalysis.supervisedTraining(path="/home/iskander/Documents/MEIOTIC_DRIVE/",
                                  crossovers="DGRP_882_129_simulations_3_19_19_crossovers.tsv")


def detect():

    myAnalysis = analyzeSEQ()
    myAnalysis.polarized_samples = myAnalysis.load_SNP_array(path='/home/iskander/Documents/MEIOTIC_DRIVE/', snp_array='polarized_simulations.npy', encoding='latin1')

    for cell in myAnalysis.polarized_samples:
        #cell_pointers = myAnalysis.hmmRBP(snp_input=cell)
        #myAnalysis.decode(cell_pointers, fileName='test.txt', path='/home/iskander/Documents/MEIOTIC_DRIVE/')
        cell_posteriors = myAnalysis.hmmFwBw(cell)
        myAnalysis.draw(cell_posteriors, cell)
        break
detect()
