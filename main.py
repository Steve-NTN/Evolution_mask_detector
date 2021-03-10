from population import Population
from individual import Individual
from load_data import mask_detect

def values(arr):
    out = sum(arr)
    return out

def fitness(p, x_train, y_train, x_test, y_test): 
    # Init fitness
    fitness_indi = 0
    N = Individual()
    p = N.individual_decode(p)
    N = N.individual_model(p, 
                        num_class=2,
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test, 
                        y_test=y_test)

    fitness_indi = N.evaluate(x=x_test, y=y_test, verbose=0)[1]
    return fitness_indi

if __name__ == "__main__":
    n_individual = 5     # Number of individual
    n_generation = 5     # Number of generation
    threshold = 0.994    # Threshold

    model = Population(num_generation=n_generation, 
                       num_individual=n_individual, 
                       score_func=fitness, 
                       prob_crossover=.8, 
                       prob_mut=.3, 
                       crossover_method='uniform',
                       threshold=threshold)
    model.fit()