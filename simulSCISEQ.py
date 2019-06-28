from __future__ import division
import os
import csv
import numpy as np
import random
import argparse
from pynverse import inversefunc
import time
import sys


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
        self.parser.add_argument('-s', '--subsample', type=str, action='store', nargs='?', default= False)
        self.parser.add_argument('-cov', '--coverage', type=int, action='store', nargs='?', default=2000)
        self.parser.add_argument('-d', '--size_distortion', type=float, action='store', nargs='?', default=0.05)
        self.parser.add_argument('-sd', '--segregation_distortion', type=float, action='store', nargs='?', default=0.05)
        self.parser.add_argument('-a', '--arm', type=int, action='store', default=np.random.randint(0, 5))
        self.parser.add_argument('-p', '--position', type=int, action='store', default=False)

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



    def createMapping(self): #Depreceated function delete soon
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

    def computeBreakpoint(self, arm, bp_max, bp_min):

        """
        In order to compute the breakpoints for our chromosome arm we call the inverse of the function that maps our BP to the cM positions.
        We are able to use the inverse of this function because it is monotonic within the domain we have defined outside of that domain it is non-recombinant so
        we don't care about those areas.

        To call a breakpoint we randomly unformly choose a cM position from a max and min range as determined by the domain of our function. Then we transform that cM into a bp position
        using the inverse of the function.

        :return:
        """
        cM_max = self.cM_map(arm)(bp_max)
        cM_min = self.cM_map(arm)(bp_min)
        cM_pos = np.random.uniform(low=cM_min, high=cM_max, size=1)

        chiasma = inversefunc(self.cM_map(arm), y_values=cM_pos)
        print(chiasma)
        chiasma = int(chiasma*1000000)
        print(cM_pos, arm, chiasma)
        return chiasma


    def read_RMAP(self): #Depreceated function delete soon
        """
        Read back in the recombination map that was computed with the cM map function

        :return:
        """
        for arm in range(5):
            self.cm_bins[arm] = {}
        with open('dmel.rmap.bed', 'r') as myMap:

            for field in csv.reader(myMap, delimiter = '\t'):
                arm = self.num_map[field[0]]
                self.cm_bins[arm][int(field[1])] = [float(field[2]), float(field[3])]
            myMap.close()

    def size_Genotype(self, genotype, simulations, geno_arm, size_distortion):
        """
        This function will generate a size dependent genotype issue by increasing the number of cells contributed by individuals with a pre-specified SNP


        To do this we can simply call a position of our size SNP and then all individuals with that SNP will have the number of cells they contributed multiplied by some factor X

        e.g. all cells homozygote at position N for a P1 allele will have X times more cells contributed
        :return:
        """


        duplicates = []

        distorted_indivs = []
        index = 0
        for cell in simulations:
            geno_block = cell[geno_arm][genotype-5:genotype+5][:,1]

            if len(set(geno_block)) == 1: #If we are a P1 homozygote for this block we will have a size increase
                distorted_indivs.append(self.simulated_crossovers[index])
            index += 1

        for sim in np.random.choice(a=len(distorted_indivs), size=size_distortion): #Randomly choose which individuals have a size distortion
            crossover = distorted_indivs[sim]
            self.sim_array = [list() for chr in range(5)]
            for arm in range(1, 6):
                self.simulateSNP(breakpoint=crossover[arm][0], reference=reference_alleles, parental=crossover[arm][1], arm=crossover[arm][2])
            self.simulated_crossovers.append(crossover)
            duplicates.append(self.sim_array)

        return duplicates


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

                        if P1_allele != np.nan and P2_allele != np.nan: #Correct for heterozygous individuals
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


    def simulateSNP(self, breakpoint, reference, parental, arm):

        #Rather than return the alleles for each homozygous or heterozygous segment we will instead simply create a pre-polarized snp array

        if len(breakpoint) == 1:#E1
            chiasma = breakpoint[0]

            if parental == 0:
                # hom segment
                segment = reference[arm][np.where(reference[arm][:, 0] <= chiasma)]
                P1 = np.zeros(shape=len(segment))
                hom_segment = np.vstack((segment[:,0], P1)).T

                #het_segment
                segment = reference[arm][np.where(reference[arm][:,0] > chiasma)]
                het = np.random.randint(0,2, size= len(segment)).astype(float)

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
                het = np.random.randint(0, 2, size=len(segment)).astype(float)
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

                het = np.random.randint(0, 2, size=len(seg_intersect)).astype(float)
                contig_2 = np.vstack((reference[arm][np.intersect1d(segment_1, segment_2)][:,0], het)).T

                #het_segment = reference[arm][np.intersect1d(segment_1, segment_2)][:, [0,1]]


                # het_segment
                segment = reference[arm][np.where(reference[arm][:, 0] > chiasma_2)]
                P1 = np.zeros(shape=len(segment))
                contig_3 = np.vstack((segment[:, 0], P1)).T

                output_parental = 0

            else:#When P1/P2 segment is first

                # het segment 1
                segment = reference[arm][np.where(reference[arm][:, 0] <= chiasma_1)]
                het = np.random.randint(0, 2, size=len(segment)).astype(float)
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
                het = np.random.randint(0, 2, size=len(segment)).astype(float)
                contig_3 = np.vstack((segment[:, 0], het)).T

                output_parental = 1

            complete_segment = np.concatenate((contig_1, contig_2, contig_3))

        elif len(breakpoint) == 0:#E0
            if parental == 0:
                P1 = np.zeros(shape=len(reference[arm][:,0]))
                complete_segment = np.vstack((reference[arm][:,0], P1)).T
                output_parental = 0
            else:
                het = np.random.randint(0, 2, size=len(reference[arm][:,0])).astype(float)
                complete_segment = np.vstack((reference[arm][:,0], het)).T
                output_parental = 1


        self.sim_array[arm] = complete_segment




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
            #data = np.argwhere(np.isnan(snp_input[chrom][:,1]) == False).T[0]
            #n = np.random.binomial(len(data), err)
            #errors = np.random.choice(data, size=n)
            #snp_input[chrom][:,1][errors] = 2


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


    def simMeiosis(self, uniq_indivs, D=.1, driver = 1):
        """
        In order to more realistically simulate drive I am going to to inherently implement a drive mechanic in my simulation of all of my individuals.
        I will do this by biasing the "meiosis" of the P1 allele in meiosis. My hope is that by doing this I will be more accurately modelling the true
        rate of allele frequency decay that would be seen in a real population.


        :return:
        """

        #Simulate the meiosis
        gamete_vector = np.zeros(shape=(3, uniq_indivs))
        for chromosome in range(3):
            if chromosome == driver: # sim meiosis with strength of driver
                gamete_vector[chromosome] = np.random.binomial(n=1, size=uniq_indivs, p=.5 - D)
            else:
                gamete_vector[chromosome] = np.random.binomial(n=1, size=uniq_indivs, p=.5)
        #Vector with gametes is now filled and we shall return it

        return gamete_vector.astype(int)

    def simulateRecomb(self, reference, gametes, simID = 1):

        self.sim_array = [list() for chr in range(5)]
        indiv_CO_inputs = [simID]

        #Generate the E value for a chromosome based on the drosophila E-values
        #E0, E1, E2
        e_values = [[[0,15], [16, 91], [92,100]], [[0,16], [17, 92], [94,100]], [[0,5], [6, 76], [77, 100]], [[0,12], [13, 79], [80, 100]], [[0,7], [8,56], [57,100]]]

        arm_to_gametes = [0,0,1,1,2] #code to translate from the arm to the full length chromosome to determine the parental gamete
        for arm in range(5):
            chromosome = arm_to_gametes[arm]
            breakpoints = []
            # Call initial state of chromosome#
            max_BP = self.heterochromatin[arm][1]
            min_BP = self.heterochromatin[arm][0]

            percentile = np.random.randint(0, 101)
            for i in range(3):
                if percentile >= e_values[arm][i][0] and percentile <= e_values[arm][i][1]:
                    E = i
                    break
                else:
                    pass

            if E == 1:#E1
                chiasma = self.computeBreakpoint(arm=arm, bp_max= max_BP, bp_min= min_BP)

                breakpoints.append(chiasma)
                #To be able to conform my gamete calling method to the simSNPs method I need to call the initial parent rather than the identity of centromeric gamete
                if arm in [1, 3]:
                    init_parent = gametes[chromosome]
                    sys.exit()
                else:
                    init_parent = abs(gametes[chromosome] - 1)

            elif E == 2:#E2

                s = 0
                while s < 10.5:
                    chiasma_1 = self.computeBreakpoint(arm=arm, bp_max= max_BP, bp_min= min_BP)
                    chiasma_2 = self.computeBreakpoint(arm=arm, bp_max= max_BP, bp_min= min_BP)
                    s = abs(chiasma_1 - chiasma_2) / 1000000
                breakpoints.append(chiasma_1)
                breakpoints.append(chiasma_2)
                init_parent = gametes[chromosome] #

            else:#E0
                init_parent = gametes[chromosome] #Entire segment is parental


            orig_p = init_parent
            self.simulateSNP(sorted(breakpoints), reference, init_parent, arm)
            CO_inputs = [sorted(breakpoints), orig_p, arm]

            sim_array = np.asarray(self.sim_array)
            indiv_CO_inputs.append(CO_inputs)
        self.simulated_crossovers.append(indiv_CO_inputs)

        return sim_array

def sample_lowCoverage(snp):
    mySimulation = simulateSEQ()
    path, file = os.path.split(snp)
    mySimulation.SNP_samples = mySimulation.load_reference(path=path, reference=file, encoding='latin1')


    snp_bounds = mySimulation.get_SNP_bounds(mySimulation.SNP_samples[0])


    for cell in range(len(mySimulation.SNP_samples)):

        cov = max(300, int(np.random.exponential(myArgs.args.coverage)))
        mySimulation.SNP_samples[cell] = mySimulation.NA_subsampler(snp_input=mySimulation.SNP_samples[cell], sampling=cov, all_subsamples=snp_bounds)

    output = os.path.join(path, 'SPARSE_'+file)
    np.save(output, mySimulation.SNP_samples)


if __name__ == '__main__':

    start = time.time()
############# Call commandline ##########

    myArgs = CommandLine()
    ###########

    if myArgs.args.subsample != False:
        sample_lowCoverage(myArgs.args.subsample)

    else:
        ########
        all_simulations = []
        simulate = simulateSEQ()

        #Read the recombination map
        simulate.read_RMAP()

        #simulate.generateSNPreference(path='/home/iskander/Documents/MEIOTIC_DRIVE/', vcf='882_129.snps.vcf')
        reference_alleles = simulate.load_reference(path=os.path.split(myArgs.args.reference)[0], reference=os.path.split(myArgs.args.reference)[1])


        #Generate several hundred recombinants with some number of multiples

        #calculate the expected number of unique cells given the cells and individual arguments:

        noisy_cells = np.random.randint(low=myArgs.args.wells*20, high = (myArgs.args.wells*25) + 1 ) #Generate the number of cells sequenced in our pool with some random noise

        E_uniq = len(set(np.random.choice(a=myArgs.args.individuals, size=noisy_cells))) #Calculate the number of unique cells by using a noisy approach through literally sampling them

        size_distortion = int(myArgs.args.size_distortion*E_uniq) # To account for a say 5% different cell contributions I will subtract 5% from the number of unique individuals

        non_Uniq = noisy_cells - E_uniq - size_distortion


        #### Generate all of the unique cells

        #Invoke meiosis first:
        gamete_vector = simulate.simMeiosis(uniq_indivs=E_uniq, D=myArgs.args.segregation_distortion)
        #Now iterate through the gametes and produce their recombination breakpoints in accordance to our allele frequencies
        for sim in range(E_uniq):
            gametes = gamete_vector[:,sim]
            simulated_SNPs = simulate.simulateRecomb(reference=reference_alleles, simID=sim+1, gametes=gametes)
            all_simulations.append(simulated_SNPs)



        #### Size distortion #####
        geno_arm = myArgs.args.arm
        if myArgs.args.position == False:
            genotype = np.random.randint(0, len(reference_alleles[geno_arm][:,0]))
        else:
            genotype = myArgs.args.position

        with open('{0}.SNP.out'.format(myArgs.args.output), 'w') as mySNP:
            mySNP.write("{0}\t{1}\n".format(simulate.chr_mapping[str(geno_arm)], reference_alleles[geno_arm][:,0][genotype]))
        mySNP.close()

        #Create a size distortion by over sampling individuals
        size_duplicates = simulate.size_Genotype(genotype=genotype, simulations=all_simulations, geno_arm= geno_arm, size_distortion= size_distortion) #Create duplicates based on a 10 SNP window around a given SNP that must be homozygous
        all_simulations = all_simulations + size_duplicates

        #Generate at random which individuals will be sampled multiple times for stochasticity
        for collision in range(non_Uniq):
            index = random.randint(0, len(simulate.simulated_crossovers)-1)
            crossover = simulate.simulated_crossovers[index]
            simulate.sim_array = [list() for chr in range(5)]

            for arm in range(1,6):
                simulate.simulateSNP(breakpoint=crossover[arm][0], reference=reference_alleles, parental= crossover[arm][1], arm=crossover[arm][2])

            simulate.simulated_crossovers.append(crossover)
            all_simulations.append(np.asarray(simulate.sim_array))

        all_simulations = np.asarray(all_simulations)

        #Save the final output of the simulations

        np.save(myArgs.args.output+'.npy', all_simulations)
        with open(myArgs.args.output+'_crossovers.tsv', 'w') as myCO:

            for ind in simulate.simulated_crossovers:
                ID = str(ind[0])
                individual = [ID]
                for arm in range(1,6):
                    #Format the breakpoints
                    bp_str = [str(bp) for bp in ind[arm][0]]
                    breakpoints= ','.join(bp_str)

                    chrom_arm = simulate.chr_mapping[str(arm-1)]

                    #Format parental haplotypes
                    parents = []
                    p_switch = {0:1, 1:0}
                    #print(chrom_arm)
                    #print(bp_str, str(ind[arm][1]))
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
