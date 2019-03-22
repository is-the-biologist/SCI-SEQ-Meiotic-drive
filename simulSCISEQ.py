import os
import csv
import numpy as np
import random




"""
A program that will simulate single cell data of an F2 cross with meiotic drivers at specified loci at varying strength.

The goal of this program is to design a simulator of low coverage reads 1-2x at any given SNP for many many individuals.



NOTE:

DGRP lines were aligned to the Release 5 genome so when we align our reads we should use release 5 OR convert our DGRP coordinates to release 6

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
            '2L': (0.53, 18.87),
            '2R': (1.87, 20.87),
            '3L':(0.75, 19.02),
            '3R':(2.58, 27.44),
            'X':(1.22, 21.21)

        }
        self.num_mapping = {0:('2L', '2R'),
                     1:('3L', '3R'),
                     2:('X')
                    }
        self.chr_mapping = {'2L':0, '2R':1, '3L':2, '3R':3, 'X':4}
        self.simulated_crossovers = []
        self.err = 0.001 #error rate from illumina sequencer
        self.sim_array = [list() for chr in range(5)]
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


    def load_reference(self, reference, path):
        np_file = os.path.join(path, reference)
        reference = np.load(np_file)

        return reference

    def simSNPs(self, breakpoint_index, reference, init_parent, crossover):
        """
        The meat and potatoes of the simulation. This function does all the heavy lifting.
        Essentially it takes in a few parameters, including the location of the crossover and generates a randomly sampled
        SNP array based on the location of the crossover, the chromosome arm, and the parental gamete.


        :param breakpoint_index:
        :param reference:
        :param init_parent:
        :param crossover:
        :return:
        """





        snp_positions = reference[breakpoint_index][:, 0]
        simulated_alleles = []
        ############### P1 #########################
        if init_parent == 1:  # When gamete is P1, P2
            for position in range(len(snp_positions)):
                error = random.uniform(0,1)
                if error <= self.err: #Introduce an error rate:
                    allele = random.randint(0,3)
                    #simulated_alleles.append([snp_positions[position], allele])
                else:
                    if snp_positions[position] >= crossover:#For one arm

                        # Once SNP position reaches crossover event we switch to the other parent to call SNPs
                        # Now all SNPs are heterozygous so we will choose parent randomly
                        allele = reference[breakpoint_index][position][random.randint(1, 2)]

                    else:

                        allele = reference[breakpoint_index][position][1]

                simulated_alleles.append([snp_positions[position], allele])

            if breakpoint_index != 4:  # Exclude the X chromosome
                simulated_alleles_2 = []# For the other arm of the chromosome
                if breakpoint_index % 2 == 0:  # If the arm being called is on the left

                    alt_arm = breakpoint_index + 1
                    right_snps = reference[alt_arm][:, 0]

                    for position in range(len(right_snps)):  # Iterate through the right arm of the chromosome and generate heterozygous allele calls
                        error = random.uniform(0, 1)
                        if error <= self.err:  # Introduce an error rate:
                            allele = random.randint(0, 3)
                            simulated_alleles_2.append([right_snps[position], allele])
                        else:
                            simulated_alleles_2.append([right_snps[position], reference[alt_arm][position][random.randint(1, 2)]])

                else:  # If the arm being called is on the right
                    # The left arm will be homozygous
                    alt_arm = breakpoint_index - 1
                    # Homozygous for P1 SNPs
                    for position in range(len(reference[alt_arm][:,0])):
                        error = random.uniform(0, 1)
                        if error <= self.err:  # Introduce an error rate:
                            allele = random.randint(0, 3)
                            simulated_alleles_2.append([reference[alt_arm][:,0][position], allele])
                        else:
                            allele = reference[alt_arm][position][1]
                            simulated_alleles_2.append([reference[alt_arm][:,0][position], allele])
            else:
                pass


###################### P2 ###########
        else:  # When gamete is P2, P1
            for position in range(len(snp_positions)):  # Get alleles of arm that had a crossover event
                error = random.uniform(0, 1)
                if error <= self.err:  # Introduce an error rate:
                    allele = random.randint(0, 3)
                    #simulated_alleles.append([snp_positions[position], allele])
                else:
                    if snp_positions[position] <= crossover:
                        # Once SNP position reaches crossover event we switch to the other parent to call SNPs
                        # Now all SNPs are heterozygous so we will choose parent randomly
                        allele = reference[breakpoint_index][position][random.randint(1, 2)]
                    else:
                        allele = reference[breakpoint_index][position][1]
                simulated_alleles.append([snp_positions[position], allele])
            # Retrieve the alleles of the arm that did not have the crossover event

            if breakpoint_index != 4:
                simulated_alleles_2 = []

                if breakpoint_index % 2 == 0:  # if arm is on the left right arm will be homozygous for P1
                    alt_arm = breakpoint_index + 1

                    for snps in range(len(reference[alt_arm][:,0])):
                        error = random.uniform(0, 1)
                        if error <= self.err:  # Introduce an error rate:
                            allele = random.randint(0, 3)
                            simulated_alleles_2.append([reference[alt_arm][:,0][snps], allele])
                        else:
                            allele = reference[alt_arm][snps][1]
                            simulated_alleles_2.append([reference[alt_arm][:, 0][snps], allele])

                else:  # in this case the left arm will be heterozygous P2/P1
                    alt_arm = breakpoint_index - 1
                    left_snps = reference[alt_arm][:, 0]

                    for position in range(len(left_snps)):
                        error = random.uniform(0, 1)
                        if error <= self.err:  # Introduce an error rate:
                            allele = random.randint(0, 3)
                            simulated_alleles_2.append([left_snps[position], allele])
                        else:
                            simulated_alleles_2.append([left_snps[position], reference[alt_arm][position][random.randint(1, 2)]])
            else:
                pass

        if breakpoint_index != 4:  # Special case for X chromosome
            # Add to our array in the correct indexing
            self.sim_array[breakpoint_index] = np.asarray(simulated_alleles)
            self.sim_array[alt_arm] = np.asarray(simulated_alleles_2)
        else:
            self.sim_array[breakpoint_index] = np.asarray(simulated_alleles)

    def simRecombinants(self, reference, simID = 1):
        """

        This function will generate simulated recombinant chromosomes for the downstream analysis.

        Take in the reference numpy arrays and randomly generate recomibination breakpoints for a new recombinant chromosome.
        We will then generate a second numpy array that will sample off of the recombinant chromosome.

        The SNPs that will appear on the recombinant chromosome are a function of which of diploid chromosomes is being sampled and how the cross was designed.

        For example:

        P1/P2 x P1 --> P1,P2/P1 || P2,P1/P1

        P1 will be the parental reference strain indexed at 1 in the reference array and P2 will be indexed at 2 in the reference array.

        In this case the part of the recombinant chromosome that is P1 homozygous will sample only P1 alleles while the part of the chromosome that would be heterozygous would sample a 50/50 mixture of P1 and P2 alleles.
        This will be modeled very simply as a uniform probability of sampling either of the alleles when the SNP calls are generated. There is also a probability of sequencing error that would behoove me to include in
        the simulation model.

        :param reference:
        :return:
        """




        ######Generate recombinant chromosomes
        self.sim_array = [list() for chr in range(5)]
        true_crossovers = [simID]
        for chromosome in range(3):
            #Call initial state of chromosome#
            init_parent = random.randint(1,2)

            #Call recombination breakpoints
            #Use a drosophila recombination map to generate a model for breakpoints

            #The probability of crossover ocurring at any given locus on the chromosome will be uniformly distributed excluding heterochromatic regions -- this not true, but is a simplifying assumpyion
            #Each crossover of each gamete will be independent of the other and I am going to exclude the possibility of a double crossover for simplicities sake
            crossovers =[]
            for arm in self.num_mapping[chromosome]:
                distance = self.heterochromatin[arm]
                crossovers.append(random.randint(distance[0]*1000000, distance[1]*1000000))
            arm = random.randint(0, len(crossovers)-1) #To simplify my code I am simply calling a crossover event to occur on each arm and then choosing at random which crossover event will happen from either arm
            crossover = crossovers[arm]



            #The chromosome now has a crossover event called on a position on an arm, and an initial gamete state (P1 or P2)
####-----------------------------replacing this with function containing important bit ---------------------#####


            #It is important to note that if our crossover event was on the left arm
            #Now we will generate our chromosome for our individual
            breakpoint_index = self.chr_mapping[self.num_mapping[chromosome][arm]]

            true_crossovers.append([self.num_mapping[chromosome][arm], breakpoint_index, init_parent, crossover])
            self.simSNPs(breakpoint_index= breakpoint_index, reference=reference, init_parent=init_parent, crossover=crossover)
            ####SNPs have been simulated for this chromosome now we add it to our numpy array###



        self.simulated_crossovers.append(true_crossovers)
        sim_array = np.asarray(self.sim_array)

        return sim_array









if __name__ == '__main__':

    all_simulations = []
    simulate = simulateSEQ()

    #simulate.generateSNPreference(path='/home/iskander/Documents/MEIOTIC_DRIVE/', vcf='882_129.snps.vcf')
    reference_alleles = simulate.load_reference(path='/home/iskander/Documents/MEIOTIC_DRIVE/', reference='882_129.snp_reference.npy')

    #Generate several hundred recombinants with some number of multiples
    for sim in range(1):
        simulated_SNPs = simulate.simRecombinants(reference=reference_alleles, simID=sim+1)
        all_simulations.append(simulated_SNPs)



    #Generate at random which individuals will be sampled multiple times
    for collision in range(1):
        index = random.randint(0, len(simulate.simulated_crossovers)-1)
        crossover = simulate.simulated_crossovers[index]
        simulate.sim_array = [list() for chr in range(5)]

        for chromosome in range(3):

            simulate.simSNPs(breakpoint_index= crossover[chromosome+1][1], reference=reference_alleles, init_parent=crossover[chromosome+1][2], crossover=crossover[chromosome+1][3])

        simulate.simulated_crossovers.append(simulate.simulated_crossovers[index])
        all_simulations.append(np.asarray(simulate.sim_array))

    all_simulations = np.asarray(all_simulations)

    test_path = os.path.join('/home/iskander/Documents/MEIOTIC_DRIVE/', 'new_test.npy')
    np.save(test_path, all_simulations)

    with open('crossovers.tsv', 'w') as myCO:
        myCO.write('ID\tCHR1\tPOS1\tP1\tCHR2\tPOS2\tP2\tCHR3\tPOS3\tP3\n')
        for lines in simulate.simulated_crossovers:
            myCO.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n".format(lines[0], lines[1][0], lines[1][3], lines[1][2],lines[2][0], lines[2][3], lines[2][2], lines[3][0], lines[3][3], lines[3][2] ))
        myCO.close()