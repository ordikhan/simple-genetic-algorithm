import random
from random import randint
import numpy as np
from DataLoader import DataParser
from sklearn.metrics import roc_curve, auc
from math import ceil

INTERVAL_BLOOD_PRESSURE = [120, 140, 160]
INTERVAL_CHOLESTEROL = [150, 200, 250, 300]


class Genetics:

    def __init__(self):
        Load = DataParser('ParsDataSet.xlsx')
        self.df = Load.dataframe
        self.pop_Row = 4
        self.pop_Column = 5
        self.list_population = []

    def CratePopulation(self, NumberOfChromosome):
        """
         :param NumberOfChromosome: Enter the number of people
         :return: Returns the set of a valid person in the population
        """
        list_population = []

        Repeat_previous_number = 0.5

        for number in range(NumberOfChromosome):
            Chromosome = [[0 for col in range(self.pop_Column)] for row in range(self.pop_Row)]
            Chromosome[0][0] = 0.1
            for col in range(1, self.pop_Column):
                while True:
                    rand = random.random()
                    if Chromosome[0][col - 1] < rand < Repeat_previous_number:
                        Chromosome[0][col] = Chromosome[0][col - 1]
                        break
                    if Chromosome[0][col - 1] < rand:
                        Chromosome[0][col] = rand
                        break

            for row in range(1, self.pop_Row):
                while True:
                    rand = random.random()
                    if Chromosome[row - 1][col] >= rand and rand < Repeat_previous_number:
                        Chromosome[row][0] = Chromosome[row - 1][0]
                        break
                    if Chromosome[row - 1][0] < rand:
                        Chromosome[row][0] = rand
                        break
                for col in range(1, self.pop_Column):
                    while True:
                        rand = random.random()
                        if Chromosome[row - 1][col] <= rand < Repeat_previous_number and rand >= Chromosome[row - 1][
                            col]:
                            Chromosome[row][col] = Chromosome[row][col - 1]
                            break
                        if Chromosome[row][col - 1] <= rand and rand >= Chromosome[row - 1][col]:
                            Chromosome[row][col] = rand
                            break

            Chromosome.reverse()
            list_population.append(Chromosome)

        return list_population

    def Fitness(self, chromosome):
        v_label = self.df.iloc[:, -1].tolist()
        v_blood_pressure = self.df.iloc[:, 14].tolist()
        v_cholesterol = self.df.iloc[:, 19].tolist()
        risk_v = []
        for index in range(len(v_label)):
            for i in range(len(INTERVAL_BLOOD_PRESSURE)):
                if v_blood_pressure[index] == INTERVAL_BLOOD_PRESSURE[i]:
                    for j in range(len(INTERVAL_CHOLESTEROL)):
                        if v_cholesterol[index] == INTERVAL_CHOLESTEROL[j]:
                            risk_v.append(chromosome[len(INTERVAL_BLOOD_PRESSURE) - (i + 1)][j])

        fpr, tpr, _ = roc_curve(v_label, risk_v)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def Select(self, population, fitness, parents_num):
        """
        :param population: One generation's population
        :param fitness: List the value of the fitting function
        :param num_parents: Number of Parents Required for Selection
        :return: Selecting the individuals in the current generation as parents for producing
         the offspring of the next generation.
        """
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

    def mate_child(self, parent1, parent2, child_num):
        """
        :param parent1: first parent
        :param parent2: second parent
        :return: Calculates all states of mating of the first parent with the second parent
        """
        child_i = self.pop_Row - ceil(child_num / self.pop_Column)
        child_j = (child_num % self.pop_Column) - 1
        if child_j == -1:
            child_j += self.pop_Column

        if child_i == 0:
            if parent2[child_i][child_j + 1] < parent1[child_i][child_j]:
                return None
        elif self.pop_Column - child_j == 1:
            if parent2[child_i - 1][child_j] < parent1[child_i][child_j]:
                return None
        elif child_i == 0 and child_j == self.pop_Column - 1:
            return None
        elif parent2[child_i][child_j + 1] < parent1[child_i][child_j] and parent2[child_i - 1][child_j] < \
                parent1[child_i][child_j]:
            return None

        child = [[0 for col in range(self.pop_Column)] for row in range(self.pop_Row)]
        current_num = 0
        for i in range(self.pop_Row - 1, 0, -1):
            for j in range(self.pop_Column):
                if current_num < child_num:
                    child[i][j] = parent2[i][j]
                    current_num += 1
                else:
                    child[i][j] = parent1[i][j]

        return child

    def mate_children(self, parent1, parent2):
        """
        :param parent1: is parent1
        :param parent2: is parent 2
        :return: result cross mating parent1 to parent 2 and vice versa parent2 to parent1
        """
        children = []

        for i in range(1, self.pop_Row * self.pop_Column - 1, 1):
            child = self.mate_child(parent1, parent2, i)
            if (child is not None):
                dup = False
                for j in range(len(children)):
                    if all(elem in children[j] for elem in child):
                        dup = True
                if not dup:
                    children.append(child)

            dup_child = self.mate_child(parent2, parent1, i)
            if (dup_child is not None):
                dup = False
                for j in range(len(children)):
                    if all(elem in children[j] for elem in dup_child):
                        dup = True
                if not dup:
                    children.append(dup_child)

        return children

    def Cross(self, parent1, parent2):
        return self.mate_children(parent1, parent2)

    def Mutation_Raw(self, chromosome, row, column):
        for index_column in range(self.pop_Column):
            if index_column < column:
                if chromosome[row][index_column] > chromosome[row][column]:
                    chromosome[row][index_column] = chromosome[row][column]
            if index_column > column:
                if chromosome[row][index_column] < chromosome[row][column]:
                    chromosome[row][index_column] = chromosome[row][column]
        return chromosome

    def Mutation_Column(self, chromosome, row, column):
        for index_row in range(self.pop_Row):
            if index_row < row:
                if chromosome[index_row][column] < chromosome[row][column]:
                    chromosome[index_row][column] = chromosome[row][column]
                    chromosome = self.Mutation_Raw(chromosome, index_row, column)
            if index_row > row:
                if chromosome[index_row][column] > chromosome[row][column]:
                    chromosome[index_row][column] = chromosome[row][column]
                    chromosome = self.Mutation_Raw(chromosome, index_row, column)
            if index_row == row:
                chromosome = self.Mutation_Raw(chromosome, index_row, column)

        return chromosome

    def Mutate(self, chromosome):
        Size_Mutation = 1
        Row = randint(0, self.pop_Row - 1)
        # select index column
        Column = randint(0, self.pop_Column - 1)
        # mutation value of index
        chromosome[Row][Column] += Size_Mutation

        chromosome = self.Mutation_Column(chromosome, Row, Column)

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
