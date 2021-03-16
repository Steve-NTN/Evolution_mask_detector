from tensorflow.keras.utils import to_categorical
from numpy.random import uniform
from individual import Individual
from load_data import mask_detect
import numpy as np
import random

x_train, x_test, y_train, y_test = mask_detect((150, 150))

class Population():
    def __init__(self, num_individual = 10, score_func = None, 
    prob_mut = 0.2, prob_crossover = .5, num_generation = 100, 
    crossover_method = 'single', verbose = True, threshold = 1):

        self.score_func = score_func
        self.n_population = num_individual
        self.prob_mut = prob_mut
        self.prob_crossover = prob_crossover
        self.population = [indiv.individual_binary() for indiv in [Individual() for _ in range(num_individual)]]
        self.scores = []
        self.num_generation = num_generation
        self.best_result = None
        self.crossover_ = crossover_method
        self.verbose = verbose
        self.threshold = threshold

    def eval_score(self):
        sc = np.array([self.score_func(i, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test) for i in self.population])
        sc[sc <= 0] = 0.0000000001
        return sc

    def verbos_func(self, best_score, best_pop):
        print('Num population:', self.n_population)
        print('Current best score:',best_score)
        print('Current best individual:', best_pop)
        print(Individual().individual_decode(best_pop))
        print("_" * 50)

    def crossover(self, one, two, method = 'single'):
        if method == 'single':
            pnt = np.random.randint(len(one))
            one_temp = np.concatenate([one[0:pnt],two[pnt:]])
            two_temp = np.concatenate([two[0:pnt], one[pnt:]])
            return one_temp, two_temp

        elif method == 'multi':
            pnt1, pnt2 = np.random.randint(len(one), size=2)
            if pnt1 > pnt2:
                pnt1, pnt2 = pnt2, pnt1
            one_temp = np.concatenate([one[0:pnt1], two[pnt1: pnt2], one[pnt2:]])
            two_temp = np.concatenate([two[0:pnt1], one[pnt1: pnt2], two[pnt2:]])
            return one_temp, two_temp
        elif method == 'uniform':
            pr = np.random.rand(len(one)) > 0.
            one_temp = one
            two_temp = two
            for i in range(len(one)):
                if pr[i]:
                    one_temp[i], two_temp[i] = two_temp[i], one_temp[i]
            return  one_temp, two_temp

    def mutation(self, one):
        temp_rand = np.random.randint(len(one))
        one[temp_rand] = 1 - one[temp_rand]
        return one

    def choose(self, scores):
        indexs = np.random.choice(np.arange(len(scores)), size = 2, p = scores/scores.sum())
        return indexs

    def inititalization(self):
        self.population = np.array([indiv.individual_binary() for indiv in [Individual() for _ in range(self.n_population)]])
        self.scores = self.eval_score()

    def print_pop(self):
        print('last population ', self.population)
        print('best result ', self.best_result)

    def fit(self):
        self.inititalization()
        best_idx = np.argsort(self.scores)[-1]
        best_score = self.scores[best_idx]
        best_pop = self.population[best_idx, :]

        for g in range(self.num_generation):
            if best_score >= self.threshold:
                self.verbos_func(best_score, best_pop)
                break
            print("Generation {}:\n".format(g+1))
            print("Individuals & Fitness: \n{}".format([i for i in zip(self.population, self.scores)]))
            new_generation = []
            for n in range(self.n_population // 2):
                # print('dfsdf', type(self.scores))
                i, j = self.choose(self.scores)

                one = self.population[i, :].copy()
                two = self.population[j, :].copy()

                if np.random.rand() < self.prob_crossover:

                    one ,two = self.crossover(one, two, self.crossover_)

                if np.random.rand() < self.prob_mut:
                    one = self.mutation(one)
                    two = self.mutation(two)

                new_generation.append(one)
                new_generation.append(two)
            
            k = np.random.choice(np.arange(len(scores)), size = 1, p = scores/scores.sum())
            three = self.population[k, :].copy()
            new_generation.append(three)

            self.population = np.array(new_generation)
            self.scores = self.eval_score()

            if np.max(self.scores) > best_score:
                best_idx = np.argsort(self.scores)[-1]
                best_score = self.scores[best_idx]
                best_pop = self.population[best_idx, :]
            else:
                worst_idx = np.argsort(self.scores)[0]
                self.population[worst_idx, :] = best_pop
                self.scores[worst_idx] = best_score

            if self.verbose:
                self.verbos_func(best_score, best_pop)
        self.best_result = best_pop