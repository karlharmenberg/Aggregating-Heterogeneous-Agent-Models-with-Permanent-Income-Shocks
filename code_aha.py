#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 09:57:32 2021

@author: karlharmenberg
"""
import numpy as np
import time

from numba import njit, int64, float64
from numba.experimental import jitclass


from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve, eigs


####################
# HELPER FUNCTIONS #
####################

@njit
def linint(x, xgrid, v):    
    if x <= xgrid[0]:
        return v[0] + (x-xgrid[0])*(v[1]-v[0])/(xgrid[1]-xgrid[0])
    if x >= xgrid[-1]:
        return v[-1] + (x-xgrid[-1])*(v[-1]-v[0])/(xgrid[-1]-xgrid[0])

    x_i = np.searchsorted(xgrid, x)
    return v[x_i-1]+(x-xgrid[x_i-1])*(v[x_i]-v[x_i-1])/(xgrid[x_i]-xgrid[x_i-1])

@njit
def random_choice(a, p):
    F = np.cumsum(p)
    
    q = np.random.uniform(0,1)

    index = np.searchsorted(F, q)
    return a[index]

@njit
def u(c, gamma):
    if gamma != 1:
        return c**(1-gamma)/(1-gamma)
    else:
        return np.log(c)

###################
# PARAMETER CLASS #
###################
parameter_data = [
    ('β', float64),
    ('σ', float64),
    ('Nm', int64),
    ('mgrid', float64[:]),
    ('Nb', int64),
    ('bgrid', float64[:]),
    ('Nz', int64),
    ('zgrid', float64[:]),
    ('Nepsilon', int64),
    ('epsilon_val', float64[:]),
    ('epsilon_prob', float64[:]),
    ('Neta', int64),
    ('eta_val', float64[:]),
    ('eta_prob', float64[:]),
    ('death_prob', float64)
]

@jitclass(parameter_data)
class problem_parameters:
    r"""
    Stores the parameter values for the problem
    """
    def __init__(self, β, σ, mgrid, bgrid, zgrid,
                 Nepsilon, epsilon_val, epsilon_prob,
                 Neta, eta_val, eta_prob,
                 death_prob):

        self.β, self.σ = β, σ
        
        self.mgrid = mgrid
        self.Nm = len(mgrid)
        self.bgrid = bgrid
        self.Nb = len(bgrid)
        self.zgrid = zgrid
        self.Nz = len(zgrid)
        
        self.Nepsilon = Nepsilon
        self.epsilon_val, self.epsilon_prob = epsilon_val, epsilon_prob
        self.Neta, self.eta_val, self.eta_prob = Neta, eta_val, eta_prob
        
        self.death_prob = death_prob

######################
# CONTINUATION VALUE #
######################

@njit
def w_compute_at_point(b_i, v, params, 
                       R, wage):

    total = 0.0
    for epsilon_i in range(params.Nepsilon):
        for eta_i in range(params.Neta):

            if R*params.bgrid[b_i]/params.eta_val[eta_i]+wage*params.epsilon_val[epsilon_i] > params.mgrid[-1]:
                total += params.β*params.epsilon_prob[epsilon_i]*\
                         params.eta_prob[eta_i]*params.eta_val[eta_i]**(1-params.σ)*(
                         v[-1])
            else:
                total += params.β*params.epsilon_prob[epsilon_i]*\
                         params.eta_prob[eta_i]*params.eta_val[eta_i]**(1-params.σ)*(
                         linint(R*params.bgrid[b_i]/params.eta_val[eta_i]+wage*params.epsilon_val[epsilon_i],
                                             params.mgrid, v))
    return total

@njit
def w_compute_array(v,params,
                    R, wage):
    w = np.zeros(params.Nb)
    
        
    for b_i in range(params.Nb):

        w[b_i] = w_compute_at_point(b_i,v,
                                   params,
                                   R, wage)
    return w

##########################
# ENDOGENOUS GRID METHOD #
##########################

@njit
def c_compute_at_point(b_i, w, params):
    
    if b_i >0 and b_i < params.Nb-1:
        w_derivative = 0.5*(w[b_i]-w[b_i-1])/(params.bgrid[b_i]-params.bgrid[b_i-1])+\
                       0.5*(w[b_i+1]-w[b_i])/(params.bgrid[b_i+1]-params.bgrid[b_i])
                       
    if b_i == 0:
        w_derivative = (w[b_i+1]-w[b_i])/(params.bgrid[b_i+1]-params.bgrid[b_i])
    if b_i == params.Nb-1:
        w_derivative = (w[b_i]-w[b_i-1])/(params.bgrid[b_i]-params.bgrid[b_i-1])
    
    ctemp = (w_derivative)**(-1/params.σ)
    mtemp = ctemp + params.bgrid[b_i]
    return ctemp, mtemp

@njit
def c_compute(w, params):
    
    c = np.zeros(params.Nb)
    mtemp_vec = np.empty(params.Nb)
    ctemp_vec = np.empty(params.Nb)
    for b_i in range(params.Nb):
        ctemp, mtemp =  c_compute_at_point(b_i, w, params)

        mtemp_vec[b_i] = mtemp
        ctemp_vec[b_i] = ctemp
        
    
    for m_i in range(params.Nm):
        if params.mgrid[m_i] < mtemp_vec[0]:
            c[m_i] = params.mgrid[m_i]
        else:
            ctemp = linint(params.mgrid[m_i], mtemp_vec, ctemp_vec)
            c[m_i] = ctemp
    return c

@njit
def egm_iteration(v_next_period, R, wage, params):
    w = w_compute_array(v_next_period,params,
                       R, wage)

    consumption_function = c_compute(w, params)
    
    return consumption_function


#####################
# TRANSITION MATRIX #
#####################

@njit
def create_transition_matrix_helper(consumption_function, weighting_scheme,
                                    params,
                                    R, wage):
    
    data = np.empty(params.Nm*params.Nepsilon*params.Neta*2*2)
    col = np.empty(params.Nm*params.Nepsilon*params.Neta*2*2)
    row = np.empty(params.Nm*params.Nepsilon*params.Neta*2*2)
    
    for epsilon_i in range(params.Nepsilon):
        for eta_i in range(params.Neta):
            for death_i in range(2):
            
                if death_i == 0:
                    prob = params.eta_prob[eta_i]*params.epsilon_prob[epsilon_i]*(1-params.death_prob)
                else:
                    prob = params.eta_prob[eta_i]*params.epsilon_prob[epsilon_i]*params.death_prob

                b_end_of_period = params.mgrid-consumption_function



                if weighting_scheme == 'None' or weighting_scheme == 'Aggregate':
                    if death_i == 0:
                        m_next_period = R*b_end_of_period/params.eta_val[eta_i]+wage*params.epsilon_val[epsilon_i]
                    else:
                        m_next_period = wage*params.epsilon_val[epsilon_i]*np.ones(params.Nm)

                if weighting_scheme == 'None':
                    weight = prob
                if weighting_scheme == 'Aggregate':
                    weight = params.eta_val[eta_i]*prob
                        
                if weighting_scheme == 'Howard':
                    weight = params.eta_val[eta_i]**(1-params.σ)*prob
                    m_next_period = R*b_end_of_period/params.eta_val[eta_i]+wage*params.epsilon_val[epsilon_i]

                
                right_index = np.maximum(np.minimum(np.searchsorted(params.mgrid, m_next_period), params.Nm-1), 1)
                left_index = right_index - 1
                
                left_weight_temp = (params.mgrid[right_index]-m_next_period)/(params.mgrid[right_index]- params.mgrid[left_index])
                left_weight = np.minimum(np.maximum(left_weight_temp,0), 1.0)
                right_weight = 1.0 - left_weight
                
                index = (params.Nm*2)*(epsilon_i + eta_i*params.Nepsilon + death_i*params.Nepsilon*params.Neta)

                
                row[index:(index+params.Nm*2)] = np.hstack((left_index, right_index))
                col[index:(index+params.Nm*2)] = np.hstack((np.arange(params.Nm), np.arange(params.Nm)))
                data[index:(index+params.Nm*2)] = np.hstack((weight*left_weight, weight*right_weight))
    
    return data, col, row

def create_transition_matrix_2d_helper(consumption_function,
                                    params,
                                    R, wage):
    
    consumption_function = np.vstack([consumption_function for k in range(params.Nz)]).transpose()
    mstate = np.vstack([params.mgrid for k in range(params.Nz)]).transpose()
    zstate = np.vstack([params.zgrid for k in range(params.Nm)])
    
    data = np.empty(params.Nm*params.Nz*params.Nepsilon*params.Neta*2*4)
    col = np.empty(params.Nm*params.Nz*params.Nepsilon*params.Neta*2*4)
    row = np.empty(params.Nm*params.Nz*params.Nepsilon*params.Neta*2*4)

    for epsilon_i in range(params.Nepsilon):
        for eta_i in range(params.Neta):
            for death_i in range(2):
            
                if death_i == 0:
                    prob = params.eta_prob[eta_i]*params.epsilon_prob[epsilon_i]*(1-params.death_prob)
                else:
                    prob = params.eta_prob[eta_i]*params.epsilon_prob[epsilon_i]*params.death_prob

                b_end_of_period = mstate-consumption_function

                if death_i == 0:
                    m_next_period = R*b_end_of_period/params.eta_val[eta_i]+wage*params.epsilon_val[epsilon_i]
                else:
                    m_next_period = wage*params.epsilon_val[epsilon_i]*np.ones((params.Nm, params.Nz))

                
                m_right_index = np.maximum(np.minimum(np.searchsorted(params.mgrid, m_next_period), params.Nm-1), 1)
                m_left_index = m_right_index - 1
                
                m_left_weight_temp = (params.mgrid[m_right_index]-m_next_period)/(params.mgrid[m_right_index]- params.mgrid[m_left_index])
                m_left_weight = np.minimum(np.maximum(m_left_weight_temp,0), 1.0)
                m_right_weight = 1.0 - m_left_weight
                
                if death_i == 0:
                    z_next_period = zstate + np.log(params.eta_val[eta_i])
                else:
                    z_next_period = 0.0

                z_right_index = np.maximum(np.minimum(np.searchsorted(params.zgrid, z_next_period), params.Nz-1), 1)
                z_left_index = z_right_index - 1
                
                z_left_weight_temp = (np.exp(params.zgrid[z_right_index])-np.exp(z_next_period))\
                                     /(np.exp(params.zgrid[z_right_index])- np.exp(params.zgrid[z_left_index]))
                z_left_weight = np.minimum(np.maximum(z_left_weight_temp,0), 1.0)
                z_right_weight = 1.0 - z_left_weight
                
                
                leftleft_index = m_left_index + z_left_index*params.Nm
                leftright_index = m_left_index + z_right_index*params.Nm
                rightleft_index = m_right_index + z_left_index*params.Nm
                rightright_index = m_right_index + z_right_index*params.Nm
                
                leftleft_weight = m_left_weight*z_left_weight
                leftright_weight = m_left_weight*z_right_weight
                rightleft_weight = m_right_weight*z_left_weight
                rightright_weight = m_right_weight*z_right_weight

                index = (params.Nm*params.Nz*4)*(epsilon_i + eta_i*params.Nepsilon + death_i*params.Nepsilon*params.Neta)
                
                row[index:(index+params.Nm*params.Nz*4)] = np.hstack((leftleft_index.flatten(order = 'F'),
                                                                      leftright_index.flatten(order = 'F'),
                                                                      rightleft_index.flatten(order = 'F'),
                                                                      rightright_index.flatten(order = 'F')))
                col[index:(index+params.Nm*params.Nz*4)] = np.hstack((np.arange(params.Nm*params.Nz), 
                                                                      np.arange(params.Nm*params.Nz),
                                                                      np.arange(params.Nm*params.Nz), 
                                                                      np.arange(params.Nm*params.Nz)))
                data[index:(index+params.Nm*params.Nz*4)] = prob*np.hstack((leftleft_weight.flatten(order = 'F'),
                                                                            leftright_weight.flatten(order = 'F'),
                                                                            rightleft_weight.flatten(order = 'F'),
                                                                            rightright_weight.flatten(order = 'F')))
        
    return data, col, row

def create_transition_matrix(consumption_function, weighting_scheme,
                             params, R, wage):
    data, col, row = create_transition_matrix_helper(consumption_function, weighting_scheme,
                                                     params,
                                                     R, wage)
    T = coo_matrix((data, (col, row)), shape = (params.Nm, params.Nm))
    return T

def create_transition_matrix_2d(consumption_function,
                             params, R, wage):
    data, col, row = create_transition_matrix_2d_helper(consumption_function,
                                                     params,
                                                     R, wage)
    T = coo_matrix((data, (col, row)), shape = (params.Nm*params.Nz, params.Nm*params.Nz))
    return T

######################
# HOWARD IMPROVEMENT #
######################

def howard_improvement_algorithm(consumption_function, transition_matrix, params):
    period_utility = u(consumption_function, params.σ)

    v = spsolve(eye(params.Nm)-params.β*transition_matrix, period_utility)
    
    return v

#######################
# OPTIMAL CONSUMPTION #
#######################

def compute_optimal_consumption_function(initial_v, R, wage, params):
    
    v = initial_v
    #print("*"*30)
    #print("Computing optimal consumption")
    error = 1.0
    iterations = 0
    while error > 1e-10:
    
        consumption_function = egm_iteration(v, R, wage, params)
            
        transition_matrix_howard = create_transition_matrix(consumption_function, 'Howard',
                                                            params, R, wage)
        
        v_new = howard_improvement_algorithm(consumption_function, transition_matrix_howard, params)
    
        
        error = np.sum(np.abs(v-v_new))
        
        #print(error)
        
        v = v_new
        
        iterations += 1
        
    print("Iterations needed= ", iterations)
        
    return consumption_function, v

###########################
# STATIONARY DISTRIBUTION #
###########################

def compute_cash_on_hand_distribution(consumption_function, params, 
                                      R, wage, which = 'Marginal'):
    
    if which == 'Marginal':
        weighting_scheme = 'None'
    if which == 'Permanent Income Weighted':
        weighting_scheme = 'Aggregate'
        
    #Write down discretized transition matrix:
    transition_matrix_aggregate \
        = create_transition_matrix(consumption_function, weighting_scheme,
                                   params, R, wage)

    #Use a scipy routine to compute the eigenvector associated with 
    #eigenvalue 1, i.e., the stationary distribution.
    eigv, stationary_distribution = \
        eigs(transition_matrix_aggregate.transpose(), k = 1, sigma = 1.0)
        
    #The eigenvector that the scipy routine returns should be 
    #normalized to sum to 1
    stationary_distribution = stationary_distribution.real.flatten()\
                              /np.sum(stationary_distribution.real)
    
    return stationary_distribution

def compute_stationary_distribution_2d(consumption_function, params, R, wage):
    transition_matrix_2d = create_transition_matrix_2d(consumption_function,
                                                           params, R, wage)

    eigv, stationary_distribution_2d = eigs(transition_matrix_2d.transpose(), k = 1, sigma = 1.0)
    stationary_distribution_2d = stationary_distribution_2d.real.flatten()/np.sum(stationary_distribution_2d.real)
    return stationary_distribution_2d.reshape((params.Nm, params.Nz), order = 'F')


###################
# IMPLIED SAVINGS #
###################

def compute_implied_savings(Kguess, initial_v, params, alpha, delta):

    R = (alpha*Kguess**(alpha-1)+(1-delta))/(1-params.death_prob)
    wage = (1-alpha)*Kguess**alpha
    inner_start = time.time()
    consumption_function, v = \
        compute_optimal_consumption_function(initial_v, R, wage, params)
    inner_mid_point = time.time()
    piw_distribution = \
        compute_cash_on_hand_distribution(consumption_function, params, 
                                          R, wage, 
                                          which = 'Permanent Income Weighted')
    inner_end = time.time()
    
    b_end_of_period = params.mgrid-consumption_function
    K = np.sum(b_end_of_period*piw_distribution)
    print("Difference:", Kguess-K)
    print("K:", K)
    print("K/Y:", K**(1-alpha))
    print("Consumption computation time =", inner_mid_point - inner_start)
    print("Distribution computation time = ", inner_end - inner_mid_point)
    print("*"*50)
    return K, v, piw_distribution

#########################
# STOCHASTIC SIMULATION #
#########################

@njit
def simulate_agent_one_period(m, P, consumption_function,
                              params,
                              R, wage,
                              distorted_probabilities):
    
    #Transitory shock
    epsilon_shock = random_choice(params.epsilon_val, params.epsilon_prob)
    
    if distorted_probabilities == False:
        #Permanent-income shock if not using the 
        #permanent-income-neutral measure
        eta_shock = random_choice(params.eta_val, 
                                  params.eta_prob)
    else:
        #If using the permanent-income-neutral measure, the shock
        #probability distribution is adjusted
        eta_shock = random_choice(params.eta_val, 
                                  params.eta_prob*params.eta_val)
    
    #Death shock
    death_shock = random_choice(np.array([0,1]), 
                                np.array([1-params.death_prob, params.death_prob]))
    
    
    if death_shock == 0:
        b = m-linint(m, params.mgrid, consumption_function)
        m_new = wage*epsilon_shock + R*b/eta_shock
        
        if distorted_probabilities == False:
            P_new = eta_shock*P
        else:
            P_new = 1.0 #With permanent-income-neutral measure, the permanent
                        #income should not be updated.
    else:
        b = 0.0
        m_new = wage*epsilon_shock
        P_new = 1.0
        
    return m_new, P_new, b

@njit
def simulate_agent_many_periods(m, P, consumption_function, Nperiods,
                               params,
                               R, wage,
                               distorted_probabilities, seed):
    
    np.random.seed(seed)
    
    mlist = np.zeros(Nperiods)
    Plist = np.zeros(Nperiods)
    blist = np.zeros(Nperiods)

    for count in range(Nperiods):
        m_new, P_new, b = simulate_agent_one_period(m, P, consumption_function,
                                                   params,
                                                   R, wage,
                                                   distorted_probabilities)
        m, P = m_new, P_new
        mlist[count] = m
        blist[count] = b
        Plist[count] = P
        
    return mlist, Plist, blist


