from __future__ import division
import os
import csv
import numpy as np
import random
import argparse
import math
import time
import sys
import pandas as pd
from multiprocessing import Pool
from functools import partial

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
            description='A simulator for single cell sequencing',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            # Allows the epilog to be formatted in the way I want!!
            epilog=('''                                      
              

              '''),
            add_help=True,  # default is True
            prefix_chars='-',
            usage='%(prog)s [options] -option1[default] <input >output'
        )  ###Add or remove optional arguments here
        self.parser.add_argument("-i", "--individuals", type=int, action="store", nargs="?", help="Input the number of individuals to be sampled from.",
                                 default=5000)
        self.parser.add_argument("-w", "--wells", type=int, action="store", nargs="?", help="The number of wells to be simulated. 20-25 cells per well.",
                                 default=96)
        self.parser.add_argument("-r", "--reference", type=str, action="store", nargs="?", help="The reference genome numpy file to draw SNPs from",
                                 default='/home/iskander/Documents/Barbash_lab/mDrive/REF/ATAC_882_129.snp_reference.npy')
        self.parser.add_argument('-o', '--output', type=str, action='store', nargs='?', help='The name of your output file', default='out')
        self.parser.add_argument('-s', '--subsample', action='store_true')
        self.parser.add_argument('-c', '--concatenate', action='store_true')
        self.parser.add_argument('-cov', '--coverage', type=int, action='store', nargs='?', default=3000)
        self.parser.add_argument('-sz', '--size_distortion', type=float, action='store', nargs='?', default=0.05)
        self.parser.add_argument('-sd', '--segregation_distortion', type=float, action='store', nargs='?', default=0.05)
        self.parser.add_argument('-dv', '--driver_chrom', type=int, action='store', nargs='?', default=1)
        self.parser.add_argument('-a', '--arm', type=int, action='store', default=np.random.randint(0, 3))
        self.parser.add_argument('-rmap', '--recombination_map', type=str, action='store', nargs='?', default='/home/iskander/Documents/Barbash_lab/mDrive/dmel.rmap.bed', help='Location of the recombination map for the genome.')
        self.parser.add_argument('-t', '--telocentric', action='store_true', help='Use this flag for D virillis simulations.')
        self.parser.add_argument('-rl', '--read_length', action='store', help='Read length to simulate sampling reads off of genome', default=75)
        self.parser.add_argument('-j', '--jobs', action='store', type=int, help='The number threads to use for this process. Only applies to sub-sampling methods.', default=1)
        self.parser.add_argument('-n', '--num_chromosomes', action="store", type=int, default=3, help='The number of whole chromosomes to segregate via meiosis only takes values 3 or 5.')

        if inOpts is None:  ## DONT REMOVE THIS. I DONT KNOW WHAT IT DOES BUT IT BREAKS IT
            self.args = self.parser.parse_args()
        else:
            self.args = self.parser.parse_args(inOpts)

"""
A program that will simulate single cell data of an F2 cross with meiotic drivers at specified loci at varying strength.

The goal of this program is to design a simulator of low coverage reads 1-2x at any given SNP for many many individuals.



NOTE:

DGRP lines were aligned to the Release 5 genome so when we align our reads we should use release 5 OR convert our DGRP coordinates to release 6
'882_129.snp_reference.npy'
"""


class simulateSEQ:
    """
    General class that will hold all our functions and such

    functions:

        init





    """
    def __init__(self):

        """

        Recombination map for drosophila was taken from the following resource:

        https://petrov.stanford.edu/cgi-bin/recombination-rates_updateR5.pl#Marey%20Maps

        """

        self.chromosomes = ('2L', '2R', '3L', '3R', 'X')
        self.heterochromatin = {
        #A map of the freely recombining intervals of euchromatin within each drosophila chromosome arm in Mb
            0: (0.53, 18.87),
            1: (1.87, 20.87),
            2:(0.75, 19.02),
            3:(2.58, 27.44),
            4:(1.22, 21.21)

        }

        self.cm_bins = {}

        self.chr_mapping = {'0':'2L', '1':'2R', '2':'3L', '3':'3R', '4':'X'}
        self.num_map = {'2L':0, '2R':1, '3L':2, '3R':3, 'X':4}
        self.simulated_crossovers = []
        self.err = 0.001 #error rate from illumina sequencer
        self.sim_array = [list() for chr in range(5)]

        self.SNP_samples = []

    def cM_map(self, arm):
        """
        The function to map a bp coordinate to cM coordinate
        Use this function to create maps

        :param x:
        :return:
        """

        chr_2L = (lambda x: -0.01*x**3 + 0.2*x**2 + 2.59*x - 1.59)
        chr_2R = (lambda x: -0.007*x**3 + 0.35*x**2 - 1.43*x + 56.91)
        chr_3L = (lambda x: -0.006*x**3 + 0.09*x**2 + 2.94*x - 2.9)
        chr_3R = (lambda x: -0.004*x**3 + 0.24*x**2 - 1.63*x + 50.26)
        chr_X = (lambda x: -0.01*x**3 + 0.30*x**2 + 1.15*x - 1.87)

        bp_to_cM = [chr_2L, chr_2R, chr_3L, chr_3R, chr_X]

        return bp_to_cM[arm]

    def createMapping(self):
        """
        Use the function to turn bp to cM to create a table to translate a cM distance to a bp window. This way we can draw cM from a random uniform distribution and in this way turn it back into a bp value

        This takes a long time to compute future updates should turn this output to a bed file and then read it in.
        :return:
        """


        for arm in range(5):
            self.cm_bins[arm] = {}
            distance = self.heterochromatin[arm]
            max_dist = int(self.cM_map(arm)(distance[1]))
            min_dist = int(max(0, self.cM_map(arm)(distance[0])))

            #Lets do 1cM long bins and place our bp values within them

            for cm in range(min_dist, max_dist+2):
                self.cm_bins[arm][cm] = []


            for bp in range(int(distance[0] * 1000000), int(distance[1] * 1000000)):
                position = bp / 1000000
                cM = max(round(self.cM_map(arm)(position)), 0)
                if len(self.cm_bins[arm][cM]) == 0:
                    self.cm_bins[arm][cM] = [position]
                elif len(self.cm_bins[arm][cM]) == 1:
                    self.cm_bins[arm][cM].append(position)
                else:
                    self.cm_bins[arm][cM][1] = position

        #Write out map into a bed file because this process takes a long ass time:
        with open('dmel.rmap.bed', 'w') as myMap:

            for arm in range(5):
                for cM in list(sorted(self.cm_bins[arm].keys())):
                    if len(self.cm_bins[arm][cM]) > 1: #check if the value got filled
                        chrom = self.chr_mapping[str(arm)]
                        myMap.write('{0}\t{1}\t{2}\t{3}\n'.format(chrom, cM, self.cm_bins[arm][cM][0], self.cm_bins[arm][cM][1]))

            myMap.close()

    def simCollisions(self):

        """
        This method will simulate true collisions where two cells share a barcode so half of the reads will come from each cell.
        In the case where two cells coincidentally got the same bar code via a random combinatorial process then that cell would look to have simply double the number of reads,
        but if there was a doublet via a sticky-ness process then the number of reads would be the same.

        This happens at a rate of 10-11% let's call it a flat 10% for simplicities sake. To simulate this we will generate two cells at random and sample SNPs off of each cell at random and merge the SNPs
        :return:
        """
        pass

    def computeBreakpoint(self, arm, chiasma_vector, E, gamete, telo=False):

        """
        In order to compute the breakpoints for our chromosome arm we call the inverse of the function that maps our BP to the cM positions.
        We are able to use the inverse of this function because it is monotonic within the domain we have defined outside of that domain it is non-recombinant so
        we don't care about those areas.

        To call a breakpoint we randomly unformly choose a cM position from a max and min range as determined by the domain of our function. Then we transform that cM into a bp position
        using the inverse of the function.


        The inverse function does not work for chr3R so I am going to have to stop implementing this.
        :return:
        """


        if E == 1:
            cM = int(np.nonzero(chiasma_vector)[0] + min(self.cm_bins[arm].keys()))
            BP_interval = self.cm_bins[arm][cM]
            chiasma = np.random.randint(low=BP_interval[0]*1000000, high=BP_interval[1]*1000000)

            # To be able to conform my gamete calling method to the simSNPs method I need to call the initial parent rather than the identity of centromeric gamete
            #Need to account for Telocentric option b/c D mel is different than D vir
            if telo == False:
                if arm in [1, 3]:
                    init_parent = gamete
                else:
                    init_parent = abs(gamete - 1)

            else: #D virillis chromosomes are ordered weird and are telocentric so they require extra finagling
                if arm in [3, 2]:
                    init_parent = gamete
                else:
                    init_parent = abs(gamete - 1)
            return [chiasma], init_parent

        elif E > 1: #Only DCO when there is 10.5 Mb distance between CO events

            init_parent = gamete
            arm_chiasma = []
            cM = np.nonzero(chiasma_vector)[0] + min(self.cm_bins[arm].keys())
            for breakpoints in cM: #Draw all breakpoint/chiasma for each recombination event in the E2,E3, etc.
                BP_interval = self.cm_bins[arm][breakpoints]
                chiasma = np.random.randint(low=BP_interval[0] * 1000000, high=BP_interval[1] * 1000000)
                arm_chiasma.append(chiasma)

            #Now we check to see if the breakpoints would be possible given crossover intereference of 10.5 Mb between breakpoints
            #Let's construct a matrix of the distances between all of the breakpoints that we called
            dist_matrix = np.zeros(shape=(E, E))
            breakpoint_matrix = np.zeros(shape=(E,E))
            for bp_1 in range(E): #compute all pairwise distances between each breakpoint
                for bp_2 in range(E):
                    distance = abs(arm_chiasma[bp_1] - arm_chiasma[bp_2]) / 1000000
                    dist_matrix[bp_1][bp_2] = distance
                    breakpoint_matrix[bp_1][bp_2] = arm_chiasma[bp_1]
            dist_max = np.where(dist_matrix >= 10.5)

            if dist_max[0].size == 2: #Obtain only the breakpoints that have a distance greater than 10.5 in the distance matrix all other breakpoints are not recorded
                output_chiasma = list(set(breakpoint_matrix[dist_max].astype(int)))

            elif dist_max[0].size > 2: #If it is a triple crossover we will ignore one of the breakpoints and just take two breakpoints that are 10.5 Mb apart
                bp = np.random.randint(low=0, high=dist_max[0].size) #choose two breakpoints at random that are greater than 10.5 Mb apart
                output_chiasma = [arm_chiasma[dist_max[0][bp]], arm_chiasma[dist_max[1][bp]]]


            elif dist_max[0].size == 0: #if none of the breakpoints are 10.5 Mb apart then we will just take one of the breakpoints at random
                output_chiasma = list(np.random.choice(arm_chiasma, size=1))


            return output_chiasma, init_parent

        else:
            return list(), gamete

    def read_RMAP(self, MAP):
        """
        Read back in the recombination map that was computed with the cM map function

        :return:
        """
        for arm in range(5):
            self.cm_bins[arm] = {}
        with open(MAP, 'r') as myMap:

            for field in csv.reader(myMap, delimiter = '\t'):
                arm = self.num_map[field[0]]
                self.cm_bins[arm][int(field[1])] = [float(field[2]), float(field[3])]
            myMap.close()

    def size_Genotype(self, cell_sampling):
        """
        This function will generate a size dependent genotype issue by increasing the number of cells contributed by individuals with a pre-specified SNP


        To do this we can simply call a position of our size SNP and then all individuals with that SNP will have the number of cells they contributed multiplied by some factor X

        e.g. all cells homozygote at position N for a P1 allele will have X times more cells contributed
        :return:
        """


        duplicates = []

        #Generate the new simulated data with the recombination breakpoints from the previous individual
        for sim in cell_sampling:
            crossover = self.simulated_crossovers[sim]
            self.sim_array = [list() for chr in range(5)]
            for arm in range(1, 6):
                self.simulateSNP(breakpoint=crossover[arm][0], reference=reference_alleles, parental=crossover[arm][1], arm=crossover[arm][2])
            self.simulated_crossovers.append(crossover)
            duplicates.append(self.sim_array)

        return duplicates

    def sizeGametes(self, size_locus, size_distortion, gametes, cell_pool, direction=0):
        """
        Individuals that have a centromere that is P1 or P2 will be designated as having size distortion. Generates a
        vector that contains which cell the individual in the gamete vector ought to be drawn from.

        The size distortion will be simulated as being a shift in the mean body size of individuals with a given locus in 100%
        LD with the centromere. This will parameterized as two normal distributions:

        N~(1, .1) and N~(1+size, .1)

        The choice of the mean as 1 is straightforward as it would make the math simpler. The SD at .1 will prevent negative
        values from being drawn most if not all the time. This parameter may need more tuning if I delve into the literature
        of Drosophila body size determinants.

        :param size_locus:
        :param size_distortion:
        :param gametes:
        :return:
        """

        total_individuals = len(gametes[size_locus])

        big_individuals = np.where(gametes[size_locus] == direction) #These individuals will have a size distortion
        normal_individuals = np.where(gametes[size_locus] != direction) #These individuals will not have size distortion

        #Draw size samples from the normal distributions respective for each population:
        big_sizes = np.random.normal(loc=1+size_distortion, scale=.1, size=len(big_individuals[0]))
        normal_sizes = np.random.normal(loc=1, scale=.1, size=len(normal_individuals[0]))

        # create a probability distribution where probability of drawing an individual is proportional to body size:
        p_distribution = np.zeros(shape=total_individuals)
        p_distribution[big_individuals] = big_sizes
        p_distribution[normal_individuals] = normal_sizes
        p_distribution = p_distribution / np.sum(p_distribution)

        #Now sample the cells in proportion to their size
        all_cells = [cell for cell in range(total_individuals)]  # array with the index of every cell
        cell_sampling = np.random.choice(a=all_cells, size=cell_pool, p=p_distribution) #Draw samples from MultiN prob distribution

        return cell_sampling

    def generateSNPreference(self, vcf, path):
        """
        We are going to construct a numpy array with the format as follows;

        [2L [Position, P1_allele, P2_allele],
        2R [Positions, P1_allele, P2_allele], etc.]

        Each numpy array within the larger array will be for each chromosome arm: 2L, 2R, 3L, 3R, and X.
        Each allele will be encoded as:

        A:0
        T:1
        C:2
        G:3


        Function will read in a VCF file for two individuals and will find all of the informative SNPs i.e. SNPs where they have distinguishable alleles.
        These alleles along with positions will be added into the numpy array described above. These P1, P2 informative SNPs will be used as a reference for
        both simulating the single cell data short reads and as a reference for downstream analysis of recombination breakpoint clustering and segregation
        distortion detection.

        :param vcf:
        :param path:
        :return:
        """


        NT_encoding = { 'A':0,
                    'T':1,
                    'C':2,
                    'G':3}
        reference_nparray = []

        scaffold_array = {}
        for chr in self.chromosomes:
            scaffold_array[chr] = []

        fullPATH = os.path.join(path, vcf)
        with open(fullPATH, 'r') as myVCF:
            vcfReader = csv.reader(myVCF, delimiter='\t')
            ##########
            next(vcfReader)#1
            next(vcfReader)
            next(vcfReader)
            next(vcfReader)
            next(vcfReader)
            next(vcfReader)
            next(vcfReader)
            next(vcfReader)
            next(vcfReader)#8 Skip the 8 header lines sorry this looks gross


            for field in vcfReader:
                #Iterate through alleles we only care when the alleles are different
                chromosome = field[0]
                position = int(field[1])
                P1 = field[9]
                P2 = field[10]
                ref_allele = field[3]
                alt_allele =field[4]

                ref_alt_encoding = {0: ref_allele, 1: alt_allele}
                #Filter for informative SNPs by looking at alleles that are distinct and by alleles that can be accurately called i.e. not ./.
                if chromosome in self.chromosomes:#Check if the scaffold we are examining is one of the ones we care to create a reference for
                    if P1 != P2 and P1 != './.' and P2 != './.':
                        #Get P1 allele
                        if P1.split('/')[0] != P1.split('/')[1]:
                            #Skip heterozygous sites for now
                            P1_allele = np.nan
                        else:
                            P1_allele = NT_encoding[ref_alt_encoding[int(P1.split('/')[0])]]#Recover the allele from the VCF ref/alt encoding and then transform into the NT encoding

                        #Get P2 allele
                        if P2.split('/')[0] != P2.split('/')[1]:
                            P2_allele = np.nan
                        else:
                            P2_allele = NT_encoding[ref_alt_encoding[int(P2.split('/')[0])]]#Recover the allele from the VCF ref/alt encoding and then transform into the NT encoding

                        if np.isnan(P1_allele) is False and np.isnan(P2_allele) is False: #Correct for heterozygous individuals
                            informative_SNP = [position, P1_allele, P2_allele]
                            scaffold_array[chromosome].append(informative_SNP)
                        else:
                            pass
                else:
                    pass

            myVCF.close()
        #Now that we have finished writing our file into a dictionary that will contain our informative SNPs for each chromosome arm/scaffold we care about we can now convert it into a numpy array to save
        #as a .np file so we can use in downstream analysis/simulation without having to reread our VCF file every time.

        for scaffold in self.chromosomes:
            reference_nparray.append(np.asarray(scaffold_array[scaffold]))

        reference_nparray = np.asarray(reference_nparray)

        outputName = os.path.join(path, '{0}.snp_reference'.format(vcf.split('.')[0]))
        np.save(outputName, reference_nparray)

    def load_reference(self, reference, path, encoding='latin1'):
        np_file = os.path.join(path, reference)
        reference = np.load(np_file, allow_pickle=True, encoding=encoding)

        return reference

    def simChiasma(self, arm):
        """
        To simulate the number of recombination events on a given arm I am going to calculate the probability of recombination event occurring
        across the chromosome arm in 1 cM bins. Then I will draw a BP position at random from these bins.

        To do this I am going to draw a binomial sampling of the size of the number of cM bins in the arm and calculate the probability of crossover in each bin.
        Essentially I am drawing a binomial sampling from a uniform probability as each bin is exactly 1 cM in length.


        :return:
        """

        P_odd = (lambda x: (1 - math.exp(-((2 * x) / 100))) / 2) #prob of CO
        chiasma_vector = np.random.binomial(n=1, size=len(self.cm_bins[arm].keys()), p=P_odd(1))
        E_value = np.sum(chiasma_vector)


        return E_value, chiasma_vector

    def simulateSNP(self, breakpoint, reference, parental, arm):

        #Rather than return the alleles for each homozygous or heterozygous segment we will instead simply create a pre-polarized snp array

        #P1 array wil be 0
        #P1/P2 array will be 1

        #When we sub sample will draw alleles from P1/P2 e.g. 0 or 1
        if len(breakpoint) == 1:#E1
            chiasma = breakpoint[0]

            if parental == 0:
                # hom segment
                segment = reference[arm][np.where(reference[arm][:, 0] <= chiasma)]
                P1 = np.zeros(shape=len(segment))
                hom_segment = np.vstack((segment[:,0], P1)).T

                #het_segment
                segment = reference[arm][np.where(reference[arm][:,0] > chiasma)]
                #het = np.random.randint(0,2, size= len(segment)).astype(float)
                het = np.full(fill_value=1, shape=len(segment)).astype(float)

                het_segment = np.vstack((segment[:,0] , het)).T

                complete_segment = np.concatenate((hom_segment, het_segment))
                output_parental = 1
            else:
                #hom segment
                segment = reference[arm][np.where(reference[arm][:, 0] > chiasma)]
                P1 = np.zeros(shape=len(segment))
                hom_segment = np.vstack((segment[:, 0], P1)).T

                # het_segment
                segment = reference[arm][np.where(reference[arm][:, 0] <= chiasma)]
                #het = np.random.randint(0, 2, size=len(segment)).astype(float)
                het = np.full(fill_value=1, shape=len(segment)).astype(float)
                het_segment = np.vstack((segment[:, 0], het)).T

                output_parental = 0
                complete_segment = np.concatenate((het_segment, hom_segment))



        elif len(breakpoint) == 2:#E2
            chiasma_1 = breakpoint[0]
            chiasma_2 = breakpoint[1]
            if parental == 0:
                # hom segment 1
                segment = reference[arm][np.where(reference[arm][:, 0] <= chiasma_1)]
                P1 = np.zeros(shape=len(segment))
                contig_1 = np.vstack((segment[:, 0], P1)).T

                #het_segment
                segment_1 = np.where(reference[arm][:, 0] <= chiasma_2)
                segment_2 = np.where(reference[arm][:, 0] > chiasma_1)
                seg_intersect = np.intersect1d(segment_1, segment_2)

                #het = np.random.randint(0, 2, size=len(seg_intersect)).astype(float)
                het = np.full(fill_value=1, shape=len(seg_intersect)).astype(float)
                contig_2 = np.vstack((reference[arm][np.intersect1d(segment_1, segment_2)][:,0], het)).T

                #het_segment = reference[arm][np.intersect1d(segment_1, segment_2)][:, [0,1]]


                # hom_segment
                segment = reference[arm][np.where(reference[arm][:, 0] > chiasma_2)]
                P1 = np.zeros(shape=len(segment))
                contig_3 = np.vstack((segment[:, 0], P1)).T

                output_parental = 0

            else:#When P1/P2 segment is first

                # het segment 1
                segment = reference[arm][np.where(reference[arm][:, 0] <= chiasma_1)]
                #het = np.random.randint(0, 2, size=len(segment)).astype(float)
                het = np.full(fill_value=1, shape=len(segment)).astype(float)
                contig_1= np.vstack((segment[:, 0], het)).T

                # hom_segment
                segment_1 = np.where(reference[arm][:, 0] <= chiasma_2)
                segment_2 = np.where(reference[arm][:, 0] > chiasma_1)
                seg_intersect = np.intersect1d(segment_1, segment_2)
                P1 = np.zeros(shape=len(seg_intersect))
                contig_2 = np.vstack((reference[arm][seg_intersect][:, 0], P1)).T

                # het_segment = reference[arm][np.intersect1d(segment_1, segment_2)][:, [0,1]]

                # het_segment_2
                segment = reference[arm][np.where(reference[arm][:, 0] > chiasma_2)]
                #het = np.random.randint(0, 2, size=len(segment)).astype(float)
                het = np.full(fill_value=1, shape=len(segment)).astype(float)
                contig_3 = np.vstack((segment[:, 0], het)).T

                output_parental = 1

            complete_segment = np.concatenate((contig_1, contig_2, contig_3))

        elif len(breakpoint) == 0:#E0
            if parental == 0:
                P1 = np.zeros(shape=len(reference[arm][:,0]))
                complete_segment = np.vstack((reference[arm][:,0], P1)).T
                output_parental = 0
            else:
                #het = np.random.randint(0, 2, size=len(reference[arm][:,0])).astype(float)
                het = np.full(fill_value=1, shape=len(reference[arm][:,0])).astype(float)
                complete_segment = np.vstack((reference[arm][:,0], het)).T
                output_parental = 1


        self.sim_array[arm] = complete_segment

    def simMeiosis(self, uniq_indivs, D, driver=1, num_chromosomes=3):
        """
        In order to more realistically simulate drive I am going to to inherently implement a drive mechanic in my simulation of all of my individuals.
        I will do this by biasing the "meiosis" of the P1 allele in meiosis. My hope is that by doing this I will be more accurately modelling the true
        rate of allele frequency decay that would be seen in a real population.


        :return:
        """

        #Simulate the meiosis

        gamete_vector = np.zeros(shape=(num_chromosomes, uniq_indivs))
        for chromosome in range(num_chromosomes):
            if chromosome == driver: # sim meiosis with strength of driver
                gamete_vector[chromosome] = np.random.binomial(n=1, size=uniq_indivs, p=.5 - D)
            else:
                gamete_vector[chromosome] = np.random.binomial(n=1, size=uniq_indivs, p=.5)

        #Vector with gametes is now filled and we shall return it

        return gamete_vector.astype(int)

    def simulateRecomb(self, reference, gametes, simID = 1, telo=False):

        self.sim_array = [list() for chr in range(5)]
        indiv_CO_inputs = [simID]

        if telo == False: #These are weird code necessary for D mel vs. D vir
            #For D melanogaster
            arm_to_gametes = [0,0,1,1,2] #code to translate from the arm to the full length chromosome to determine the parental gamete
        else:
            #For D virillis
            arm_to_gametes = [x for x in range(5)]
        for arm in range(5):
            chromosome = arm_to_gametes[arm]
            breakpoints = []
            # Call initial state of chromosome#

            # Generate the E value for a chromosome based on the drosophila E-values
            # E0, E1, E2
            E, chiasma_vector = self.simChiasma(arm)
            chiasma, init_parent = self.computeBreakpoint(arm=arm, chiasma_vector=chiasma_vector, E=E, gamete=gametes[chromosome], telo=telo)

            breakpoints = breakpoints + chiasma

            orig_p = init_parent

            self.simulateSNP(sorted(breakpoints), reference, init_parent, arm)
            CO_inputs = [sorted(breakpoints), orig_p, arm]

            sim_array = np.asarray(self.sim_array)
            indiv_CO_inputs.append(CO_inputs)
        self.simulated_crossovers.append(indiv_CO_inputs)

        return sim_array

    def NA_subsampler(self, snp_input, sampling, all_subsamples):



        err = 0.001

        genome_pos = np.concatenate((snp_input[0][:, 0], snp_input[1][:, 0], snp_input[2][:, 0], snp_input[3][:, 0], snp_input[4][:, 0]))

        # generate random SNPs to become NaNs
        rand_snps = sorted(random.sample(range(len(genome_pos) - 1), len(genome_pos) - (int(sampling))))

        for chrom in range(5):

            # Filled the SNP array with NaNs
            for snp in rand_snps:
                if snp >= all_subsamples[chrom][0] and snp <= all_subsamples[chrom][1]:
                    translate_snp = snp - all_subsamples[chrom][0]

                    snp_input[chrom][translate_snp][1] = np.nan

                elif snp > all_subsamples[chrom][1]:
                    break

            # Add sequencing errors to the SNP arrays
            data = np.argwhere(np.isnan(snp_input[chrom][:,1]) == False).T[0]
            n = np.random.binomial(len(data), err)
            errors = np.random.choice(data, size=n, replace=False)
            snp_input[chrom][:,1][errors] = 2


        return snp_input

    def get_SNP_bounds(self, snp_input):
        # Subroutine for finding indices of SNP boundaries
        zero_pos = 0
        snp_index = []
        for chrom in range(5):
            end_index = len(snp_input[chrom]) - 1
            snp_index.append([zero_pos, end_index + zero_pos])

            zero_pos += end_index + 1

        return snp_index

class modData:

    """
    This class contains seperate functions from the simulator that are used to concatenate the Dmel numpy arrays and correct
    coordinates due to the weird way that the Dmel genome is annotated.


    """
    def __init__(self):
        self.SNP_samples = []
        self.genome_intervals = None
        self.chroms = None
        self.err = 0.001

    def readChromatin(self, acessible_sites='/home/iskander/Documents/Barbash_lab/mDrive/dmel_ATAC_peaks.bed'):
        """
        Function reads in a bed file that contains the genomic intervals that reads can be sampled off of.
        :return:
        """

        self.genome_intervals = pd.read_csv(acessible_sites, sep='\t', header=None)

        seen_chroms = set()
        add_chroms = seen_chroms.add
        self.chroms = np.asarray([x for x in self.genome_intervals.values[:, 0] if not (x in seen_chroms or add_chroms(x))] ) # add the chromosome names in order from the rmap file

    def drawReads(self, cell_cov, readlength):
        """
        Now we sample reads off of the SNP array by drawing 75bp long reads from the acessible genomic intervals as determined
        from the readChromatin function

        :param cell:
        :param coverage:
        :return:
        """

        #to make the function ammenable to multiprocessing we have to zip our data with coverage and then unpack it within the function
        cell = cell_cov[0]
        coverage = cell_cov[1]

        mappedReads = [[] for x in range(len(cell))]
        reads = np.random.choice(a=self.genome_intervals.values.shape[0], size=coverage) #draw from the genomic intervals

        tot_snps = 0
        no_snps = 0
        for read in reads: #now simulate drawing a read from within the genomic interval
            genome_interval = [self.genome_intervals.values[read][1] + int(readlength/2), self.genome_intervals.values[read][2] - int(readlength/2)]
            chromosome = str(self.genome_intervals.values[read][0])

            #draw a read from the interval:

            pos = np.random.randint(low=genome_interval[0], high=genome_interval[1]+1)
            chrom_index = np.where(self.chroms == chromosome)[0][0]
            chromSNPs = cell[chrom_index][:,0] #Get the SNP array
            snps_sampled = list(np.intersect1d(np.where(chromSNPs >= pos-int(readlength/2)), np.where(chromSNPs <= pos + int(readlength/2))) )#retrieve all of the SNPs within the 75 bp interval
            tot_snps += len(snps_sampled)
            #retain which SNPs were sampled by which read so I can be fancy and do over sampling of certain genotypes
            if len(snps_sampled) > 0:
                mappedReads[chrom_index].append(snps_sampled)
            else:
                no_snps += 1

        #Now reads have been drawn and the indices of each SNP are in arrays I can fill out an array:

        lowCov_SNPs = []
        for chromosome in range(len(mappedReads)):
            SNP_calls = np.full(fill_value=np.nan, shape=cell[chromosome].shape[0])
            for SNPs in mappedReads[chromosome]:
                SNP_state = cell[chromosome][SNPs][:,1]
                het_call = np.random.randint(low=0, high=2) #randomly choose P1 or P2 for a het site, but be the same for all on the same read
                SNP_state[np.where(SNP_state == 1)] = het_call

                SNP_calls[SNPs] = SNP_state #replace our NaN filled array with our SNP calls

            #Now let's add error into the sub sampling:
            called = np.where(np.isnan(SNP_calls) == False) #Find all non-empty sites

            err_sampling = np.random.binomial(p=self.err, n=1, size=len(SNP_calls[called])) #compute bernoulli random trials w/ sequencing error as the p
            #if the trial is success i.e. a 1 that means that we have an error we then sample three states at random w/ weighted probabilities:

            # 0 -- P1 25% chance
            # 1 -- P2 25% chance
            # 2 -- neither P1 nor P2 50% chance
            err_snps = np.random.choice(a=[0, 1, 2], p=[.25, .25, .5], size=len(err_sampling[np.where(err_sampling == 1)]))
            SNP_calls[called[0][np.where(err_sampling == 1)]] = err_snps #now replace all the SNPs that were called as errors with the new value
            chrom_array = np.column_stack((cell[chromosome][:,0], SNP_calls))
            lowCov_SNPs.append(chrom_array)

        #Now all the chromosomes have been sub sampled given the reads we can output it from this function and recurr for all individuals

        return np.asarray(lowCov_SNPs)

    def concatArrays(self, SNP):
        """
        Function to simply concatenate L and R arms of the chromosomes in the melanogaster assembly

        :param SNP:
        :return:
        """
        concatSNPs = []
        SNP[1][:,0] = SNP[1][:,0] + 23000000
        SNP[3][:,0] = SNP[3][:,0] + 24500000

        concatSNPs.append(np.vstack((SNP[0], SNP[1])) )
        concatSNPs.append(np.vstack((SNP[2], SNP[3])))
        concatSNPs.append(SNP[4])

        return np.asarray(concatSNPs)

    def NA_subsampler(self, snp_input, sampling, all_subsamples):



        err = 0.001

        genome_pos = np.concatenate((snp_input[0][:, 0], snp_input[1][:, 0], snp_input[2][:, 0], snp_input[3][:, 0], snp_input[4][:, 0]))

        # generate random SNPs to become NaNs
        rand_snps = sorted(random.sample(range(len(genome_pos) - 1), len(genome_pos) - (int(sampling))))

        for chrom in range(5):

            # Filled the SNP array with NaNs
            for snp in rand_snps:
                if snp >= all_subsamples[chrom][0] and snp <= all_subsamples[chrom][1]:
                    translate_snp = snp - all_subsamples[chrom][0]

                    snp_input[chrom][translate_snp][1] = np.nan

                elif snp > all_subsamples[chrom][1]:
                    break

            # Add sequencing errors to the SNP arrays
            data = np.argwhere(np.isnan(snp_input[chrom][:,1]) == False).T[0]
            n = np.random.binomial(len(data), err)
            errors = np.random.choice(data, size=n, replace=False)
            snp_input[chrom][:,1][errors] = 2


        return snp_input

    def get_SNP_bounds(self, snp_input):
        # Subroutine for finding indices of SNP boundaries
        zero_pos = 0
        snp_index = []
        for chrom in range(5):
            end_index = len(snp_input[chrom]) - 1
            snp_index.append([zero_pos, end_index + zero_pos])

            zero_pos += end_index + 1

        return snp_index


def simSR():
    """
    Wrapper function to simulate short reads sampled off of each cell

    :return:
    """

    myMods = modData()
    mySimulation = simulateSEQ()
    myMods.readChromatin()


    path, file = os.path.split(myArgs.args.reference)
    myMods.SNP_samples = mySimulation.load_reference(path=path, reference=file, encoding='latin1')


    coverage = np.random.geometric(p= (1/myArgs.args.coverage), size=len(myMods.SNP_samples))  # Draw number of reads from a geometric distribution
    covArgs = zip(myMods.SNP_samples, coverage) #We zip our SNP arrays and the coverage that they will get into an iterable
    #because multiprocessing.map can only take one iterable as an argument

    with Pool(processes=myArgs.args.jobs) as myPool:
        lowCov_SNPs = myPool.map(partial(myMods.drawReads, readlength=myArgs.args.read_length), covArgs)
        myPool.close()

    output = os.path.join(path, 'SPARSE_'+file)
    np.save(output, np.asarray(lowCov_SNPs))


def concat_dMel():
    SNP_data = []
    myMods = modData()
    mySimulation = simulateSEQ()
    path, file = os.path.split(myArgs.args.reference)
    myMods.SNP_samples = mySimulation.load_reference(path=path, reference=file, encoding='latin1')
    for cell in myMods.SNP_samples:
        concat_SNPs = myMods.concatArrays(SNP=cell)
        SNP_data.append(concat_SNPs)

    SNP_data = np.asarray(SNP_data)

    output = os.path.join(path, file[:-4] + '.concat.npy')
    np.save(output, SNP_data)


def sample_lowCoverage(snp):
    myMods = modData()
    mySimulation = simulateSEQ()
    path, file = os.path.split(snp)
    myMods.SNP_samples = mySimulation.load_reference(path=path, reference=file, encoding='latin1')


    snp_bounds = myMods.get_SNP_bounds(mySimulation.SNP_samples[0])


    for cell in range(len(mySimulation.SNP_samples)):

        cov = max(300, int(np.random.exponential(myArgs.args.coverage)))
        mySimulation.SNP_samples[cell] = myMods.NA_subsampler(snp_input=mySimulation.SNP_samples[cell], sampling=cov, all_subsamples=snp_bounds)

    output = os.path.join(path, 'SPARSE_'+file)
    np.save(output, mySimulation.SNP_samples)


if __name__ == '__main__':

    start = time.time()
############# Call commandline ##########

    myArgs = CommandLine()
    ###########

    if myArgs.args.subsample == True:
        simSR()

    elif myArgs.args.concatenate == True:
        concat_dMel()

    else:
        ########
        all_simulations = []
        simulate = simulateSEQ()

        #Read the recombination map
        simulate.read_RMAP(MAP=myArgs.args.recombination_map)

        #simulate.generateSNPreference(path='/home/iskander/Documents/MEIOTIC_DRIVE/', vcf='882_129.snps.vcf')
        reference_alleles = simulate.load_reference(path=os.path.split(myArgs.args.reference)[0], reference=os.path.split(myArgs.args.reference)[1])


        #Generate several thousand recombinants with some number of multiples

        # Invoke meiosis first:
        gamete_vector = simulate.simMeiosis(uniq_indivs=myArgs.args.individuals, D=myArgs.args.segregation_distortion,
                                            driver=myArgs.args.driver_chrom, num_chromosomes=myArgs.args.num_chromosomes)

        #The size distortion will increase the probability of drawing individuals from the gamete vector proportional to the distortion parameter
        noisy_cells = np.random.randint(low=myArgs.args.wells * 20, high=(myArgs.args.wells * 25) + 1)  # Generate the number of cells sequenced in our pool with some random noise
        cell_samples = simulate.sizeGametes(size_locus=myArgs.args.arm, size_distortion=myArgs.args.size_distortion,
                                            cell_pool=noisy_cells, gametes=gamete_vector)

        #We now have a vector containing all of identities of all cells. Now must be a little tricky to conform this data structure to our other methods
        E_uniq = list(set(cell_samples))


        #### Generate all of the unique cells

        #Now iterate through the gametes and produce their recombination breakpoints in accordance to our allele frequencies

        index = 0
        for sim in E_uniq:
            gametes = gamete_vector[:,sim]
            simulated_SNPs = simulate.simulateRecomb(reference=reference_alleles, simID=index, gametes=gametes, telo=myArgs.args.telocentric)
            all_simulations.append(simulated_SNPs)
            index += 1


        #### Size distortion #####

        #First must get all of the duplicate indices from our cell_sampling array:
        Non_uniq = []
        index = 0
        for cell in E_uniq:
            duplis = len(cell_samples[np.where(cell_samples == cell)][:-1])
            simulated_duplicates = np.full(shape=duplis, fill_value=index)
            if len(Non_uniq) == 0:
                Non_uniq = simulated_duplicates
            else:
                Non_uniq = np.concatenate((Non_uniq, simulated_duplicates))
            index += 1

        #Now have the list of all of our duplicates we can supply them to our size_duplicates method
        size_duplicates = simulate.size_Genotype(cell_sampling=Non_uniq) #Create duplicates based on a 10 SNP window around a given SNP that must be homozygous
        all_simulations = all_simulations + size_duplicates

        #Write out SNP file that contains positional info of our segregation distortion
        if myArgs.args.telocentric == True:
            chroms = {0:'2', 1:'3', 2:'4', 3:'5', 4:'X'}
        else:
            chroms = {0:'2L', 1:'2R', 2:'3L',  3:'3R', 4:'X'}
        with open('{0}.log'.format(myArgs.args.output), 'w') as mySNP:
            mySNP.write("SizeDistortion\tChrom:{0}\tStrength:{1}\tPos:CENTROMERE\n".format(chroms[myArgs.args.arm], myArgs.args.size_distortion))
            mySNP.write("SegDistortion\tChrom:{0}\tStrength:{1}\tPos:CENTROMERE\n".format(chroms[myArgs.args.driver_chrom], myArgs.args.segregation_distortion))
            mySNP.write('{0} individuals\t{1} cells'.format(myArgs.args.individuals, noisy_cells))
        mySNP.close()



        np.save(myArgs.args.output+'.npy', all_simulations)
        with open(myArgs.args.output+'_crossovers.tsv', 'w') as myCO:

            for ind in simulate.simulated_crossovers:
                ID = str(ind[0])
                individual = [ID]
                for arm in range(1,6):
                    #Format the breakpoints
                    bp_str = [str(bp) for bp in ind[arm][0]]
                    breakpoints = ','.join(bp_str)

                    chrom_arm = chroms[arm-1]

                    #Format parental haplotypes
                    parents = []
                    p_switch = {0:1, 1:0}

                    for p in range(len(bp_str)+1):
                        if p % 2 == 0:
                            p_state = str(ind[arm][1])
                        else:
                            p_state = str(p_switch[ind[arm][1]])
                        #print(p_state)
                        parents.append(p_state)
                    parents = ','.join(parents)
                    field = [chrom_arm, breakpoints, parents]

                    individual = individual + field
                line = '\t'.join(individual)
                myCO.write(line+'\n')

            myCO.close()
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    sys.stdout.write("Time elapsed: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
