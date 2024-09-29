import numpy as np
from itertools import product
'''
Implementation of Sum product variable elimination algorithm from scratch.
'''
def generate_combinations(n):
    """generate combinations of 0 and 1 for n variables"""
    combinations = list(product([0, 1], repeat=n))
    return combinations

class Factor:
    def __init__(self, scope: list[str], values: np.ndarray):
        self.scope = sorted(scope)
        self.values = values

    def sum_out(self, variable: str):
        """sum out the given variable and return new factor without it"""
        index = self.scope.index(variable) # find the index of the variable to sum out
        new_values = np.sum(self.values, axis=index) # sum along the axis of that var
        new_scope = [var for var in self.scope if var != variable] # remove it from the scope

        return Factor(new_scope, new_values)

    def __mul__(self, other):
        """multiply the 2 given factors and return the new factor"""

        combined_scope = sorted(set(self.scope) | set(other.scope))
        n = len(combined_scope)

        combinations  = generate_combinations(len(combined_scope))
        #print(combinations)

        array = np.ones(2**n)
    
        # Reshape the array to have dimensions 2x2x...x2 (n times)
        array = array.reshape(tuple([2] * n))

        result = Factor(combined_scope, array)

        # example . factor 1 has 'A','B',  factor 2 has 'B','C'. 
        # combined = [ 'A','B','C' ]
        # assume combination 
        # [ 0, 1, 0] maps to
        # [ A, B, C],  meaning A = 0, B=1, C=0
        # for factor 1 we need [0][1], for factor 2 we need [1][0]
        # selector:

        selector_self = [variable in self.scope for variable in combined_scope]
        selector_other = [variable in other.scope for variable in combined_scope]

        #print(selector_self) # in our case must be [true, true, false]
        #print(selector_other) # in our case must be [false, true, true]

        for combination in combinations:
            # goal: combination [0,1,0] -> result[0,1,0] = factor1[0,1] * factor2[1,0]
            self_indices = tuple(combination[i] for i in range(len(combination)) if selector_self[i])
            other_indices = tuple(combination[i] for i in range(len(combination)) if selector_other[i])
            
            fac1 = self.values[self_indices]
            fac2 = other.values[other_indices]

            #print(combination)
            #print(fac1, fac2)

            result.values[tuple(combination)] = fac1 * fac2


        #print(combined_scope)
        #print(result.values)
        return result

# Algorithm 9.1
def sum_product_VE(factors: list[Factor], vars_eliminate: list[str]):
    for variable in vars_eliminate: # ordering
        factors = sum_product_eliminate_var(factors, variable)
    
    factor_star = factors[0]  # initialize with the first factor
    for factor in factors[1:]:
        factor_star *= factor
    
    return factor_star

def sum_product_eliminate_var(factors, variable_elim):
    factors_1 = [factor for factor in factors if variable_elim in factor.scope]
    factors_2 = [factor for factor in factors if factor not in factors_1]

    psi = factors_1[0]  # initialize with the first factor
    for factor in factors_1[1:]:
        psi *= factor

    tau = psi.sum_out(variable_elim)

    factors_2.append(tau)
    return factors_2

# factors given in exercise 6

#
array = np.array([.2, .8])
f1 = Factor(['A'], array)

#
array = np.array([
    [.3, .7],
    [.5, .5]])
f2 = Factor(['A','B'], array)

#
array = np.array([
    [.8, .2],
    [.2, .8]])
f3 = Factor(['A','C'], array)

#
array = np.ones((2,2,2))
array[0,0,0] = .7  # array[B=b,C=c,D=d]
array[0,0,1] = .3
array[0,1,0] = .6
array[0,1,1] = .4
array[1,0,0] = .4
array[1,0,1] = .6
array[1,1,0] = .1
array[1,1,1] = .9
f4 = Factor(['B','C','D'], array)

#
array = np.array([
    [.9, .1],
    [.1, .9]
])
f5 = Factor(['D','E'], array)

#
array = np.array([
    [.2, .8],
    [.6, .4]
])
f6 = Factor(['C','F'], array)

def main():

    # first run
    eliminated_factor = sum_product_VE(factors=[f1, f2, f3, f4, f5, f6], vars_eliminate=['A','F','B','C'])

    print("scope of new factor after elimination:", eliminated_factor.scope)
    print("values of factor:\n", eliminated_factor.values)

    total_product : Factor = f1 * f2 * f3 * f4 * f5 * f6
    partition = np.sum(total_product.values)
    normalized_elim_factor = Factor(eliminated_factor.scope, eliminated_factor.values/partition)
    print("values of factor after division by the partition function:\n", normalized_elim_factor.values)
    print("notice that its already normalized because we started with probabilities.")
    c =  1/ (normalized_elim_factor.values[0,1] + normalized_elim_factor.values[1,1])
    print("find c so that P(D = 0 | E=1) + P(D = 1 | E=1) = 1 :", c)

    print('\n======================================== Answer:\n')
    print("\tP(D = 0 | E=1) = ", c * normalized_elim_factor.values[0,1])
    print("\tP(D = 1 | E=1) = ", c * normalized_elim_factor.values[1,1])
    print('\n===================================================')

    # check using  bayes rule

    # finding P(E=1)
    eliminated_factor = sum_product_VE(factors=[f1, f2, f3, f4, f5, f6], vars_eliminate=['A','F','B','C','D'])

    print("scope of new factor after elimination:", eliminated_factor.scope)
    print("values of factor:\n", eliminated_factor.values)

    total_product : Factor = f1 * f2 * f3 * f4 * f5 * f6
    partition = np.sum(total_product.values)
    normalized_elim_factor = Factor(eliminated_factor.scope, eliminated_factor.values/partition)
    print("values of factor after division by the partition function:\n", normalized_elim_factor.values)
    print("notice that its already normalized because we started with probabilities.")
    
    p_E1 = normalized_elim_factor.values[1]

    # finding P(D=1)
    eliminated_factor = sum_product_VE(factors=[f1, f2, f3, f4, f5, f6], vars_eliminate=['A','F','B','C','E'])

    print("scope of new factor after elimination:", eliminated_factor.scope)
    print("values of factor:\n", eliminated_factor.values)

    total_product : Factor = f1 * f2 * f3 * f4 * f5 * f6
    partition = np.sum(total_product.values)
    normalized_elim_factor = Factor(eliminated_factor.scope, eliminated_factor.values/partition)
    print("values of factor after division by the partition function:\n", normalized_elim_factor.values)
    print("notice that its already normalized because we started with probabilities.")
    p_D1 = normalized_elim_factor.values[1]

    # bayes rule
    print('\n====================== Results with Bayes Rule:\n')
    print("P(E=1) =", p_E1)
    print("P(D=1) = ", p_D1)
    print('result with bayes rule = ', f5.values[1,1] * p_D1 / p_E1)
    print('===============================\n')

main()
