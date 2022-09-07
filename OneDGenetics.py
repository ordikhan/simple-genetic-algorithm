import random
import numpy as np
from DataLoader import DataParser
from sklearn.metrics import roc_curve, auc


INTERVAL_BLOOD_PRESSURE = [1, 2, 3, 4]


class Genetics:

    def __init__(self):
        Load = DataParser('ParsDataSet.xlsx')
        self.df = Load.dataframe
        self.Repeat_previous_number = 0.5
        self.v_label = self.df.iloc[:, -1].tolist()
        self.v_blood_pressure = self.df.iloc[:, 14].tolist()

    def CratePopulation(self, numberOfChromosome):
        list_population = []
        for Chromosome in range(numberOfChromosome):
            Chromosome = [0 for i in range(len(INTERVAL_BLOOD_PRESSURE))]
            Chromosome[0] = 0.1

            for index in range(1, len(INTERVAL_BLOOD_PRESSURE)):
                while True:
                    rand = random.random()
                    if Chromosome[index - 1] >= rand and rand < self.Repeat_previous_number:
                        Chromosome[index] = Chromosome[index - 1]
                        break
                    elif Chromosome[index - 1] < rand:
                        Chromosome[index] = rand
                        break

            list_population.append(Chromosome)

        return list_population

    def Fitness(self, chromosome):
        risk_v = []
        for i in range(len(self.v_label)):
            for j in range(len(INTERVAL_BLOOD_PRESSURE)):
                if self.v_blood_pressure[i] == INTERVAL_BLOOD_PRESSURE[j]:
                    risk_v.append(chromosome[j])
        fpr, tpr, _ = roc_curve(self.v_label, risk_v)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def Select(self, population, fitness, parents_num):

        total_fitness = sum(fitness)
        possibility_fitness = [fit / total_fitness for fit in fitness]
        cumulative_probability = [sum(possibility_fitness[:index_ind + 1]) for index_ind in
                                  range(len(possibility_fitness))]

        list_parents = []
        # Draw new population
        for n in range(parents_num):
            random_number = random.random()
            for (index_population, person) in enumerate(population):
                if random_number <= cumulative_probability[index_population]:
                    list_parents.append(person)
                    break
        return list_parents

    def Cross(self, parent1, parent2):

        child = []
        for i in range(1, len(parent1)):
            child.append(parent2[:i] + parent1[i:])
            child.append(parent1[:i] + parent2[i:])
        return child

    def Mutate(self, chromosome):
        Size_Mutation = 0.1
        RandomIndex = random.randint(0, len(chromosome))
        chromosome[RandomIndex] += Size_Mutation
        for index_row in range(len(chromosome)):
            if index_row < RandomIndex:
                if chromosome[index_row] < chromosome[RandomIndex]:
                    chromosome[index_row] = chromosome[index_row]
            if index_row > RandomIndex:
                if chromosome[index_row] >= chromosome[RandomIndex]:
                    chromosome[index_row] = chromosome[RandomIndex]
        return chromosome

    def GenerateRace(self, PopulationSize, NumberGenerations, PercentageChangePopulation, PercentageMutation):

        avg_pop = []
        min_pop = []
        max_pop = []
        pop = self.CratePopulation(PopulationSize)

        for ite in range(NumberGenerations):
            new_pop = []
            fitness = []
            # Calculate population fitness function
            for person in pop:
                fitness.append(self.Fitness(person))
            # how to learn
            min_pop.append(min(fitness))
            max_pop.append(max(fitness))
            avg_pop.append((sum(fitness)) / len(fitness))

            # Parental choice for reproduction
            for index in range(int(PopulationSize / 2)):

                children = self.Cross(self.Select(pop, fitness, 2)[0], self.Select(pop, fitness, 2)[1])
                new_fitness = []
                for child in children:
                    new_fitness.append(self.Fitness(child))
                if PercentageMutation > np.random.rand():
                    self.Mutate(self.Select(children, new_fitness, 2)[0])
                if PercentageMutation > np.random.rand():
                    self.Mutate(self.Select(children, new_fitness, 2)[1])
                new_pop.append(self.Select(children, new_fitness, 2)[0])
                new_pop.append(self.Select(children, new_fitness, 2)[1])

            # Replacement number of two populations
            new_pop_select_size = int(PopulationSize * PercentageChangePopulation)
            old_pop_select_size = PopulationSize - new_pop_select_size
            new_selected = self.Select(new_pop, new_fitness, new_pop_select_size)
            old_selected = self.Select(pop, fitness, old_pop_select_size)
            new_selected.extend(old_selected)
            pop = new_selected
            return pop
