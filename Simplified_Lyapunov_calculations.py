# Import numpy
import numpy as np
import scipy as sci
import scipy.integrate
from scipy import stats
from multiprocessing import Pool, Value

trials = 10000
years = 400

# Define universal gravitation constant
G = 6.67408e-11  # N-m2/kg2
# Reference quantities
m_nd = 1.989e+30  # kg #the solar mass (M☉)
r_nd = 35.6 * 1.496e+11  # m #35.6 times the astronomical unit (AU)
v_nd = 30000
t_nd = 365 * 24 * 3600  # s #one year in seconds
# Net constants
K1 = G * t_nd * m_nd / (r_nd**2 * v_nd)
K2 = v_nd * t_nd / r_nd

# Define masses (solar mass)
m1 = 1
m2 = 1
m3 = 1

# Define initial velocities
r1_i = slice(None, 3)
r2_i = slice(3, 6)
r3_i = slice(6, 9)
v1_i = slice(9, 12)
v2_i = slice(12, 15)
v3_i = slice(15, 18)

dr1bydt_i = slice(None, 3)
dr2bydt_i = slice(3, 6)
dr3bydt_i = slice(6, 9)
dv1bydt_i = slice(9, 12)
dv2bydt_i = slice(12, 15)
dv3bydt_i = slice(15, 18)

# define time span
time_span = np.linspace(0, years,
                        100 * years)  #20 orbital periods and 500 points

delta_r = 1 * (10**(-10))


def ThreeBodyEquations(w, t, results):
    r12 = sci.linalg.norm(w[r2_i] - w[r1_i])
    r13 = sci.linalg.norm(w[r3_i] - w[r1_i])
    r23 = sci.linalg.norm(w[r3_i] - w[r2_i])

    results[dr1bydt_i] = K2 * w[v1_i]
    results[dr2bydt_i] = K2 * w[v2_i]
    results[dr3bydt_i] = K2 * w[v3_i]

    results[dv1bydt_i] = K1 * m2 * (w[r2_i] - w[r1_i]) / r12**3 + K1 * m3 * (
        w[r3_i] - w[r1_i]) / r13**3
    results[dv2bydt_i] = K1 * m1 * (w[r1_i] - w[r2_i]) / r12**3 + K1 * m3 * (
        w[r3_i] - w[r2_i]) / r23**3
    results[dv3bydt_i] = K1 * m1 * (w[r1_i] - w[r3_i]) / r13**3 + K1 * m2 * (
        w[r2_i] - w[r3_i]) / r23**3

    return results


def main(trial_i):

    w = np.array([
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01,
        0.01,
        0,
        -0.05,
        0,
        -0.1,
        0,
        -0.01,
        0,
    ])
    results = np.empty((18, ))

    # Define initial position vectors
    w[:9] = np.random.uniform(0.5, 1, size=9)

    # Find Centre of Mass
    r_com = (m1 * w[r1_i] + m2 * w[r2_i] + m3 * w[r3_i]) / (m1 + m2 + m3)

    # Find velocity of COM
    v_com = (m1 * w[v1_i] + m2 * w[v2_i] + m3 * w[v3_i]) / (m1 + m2 + m3)

    in_three_body_sol = sci.integrate.odeint(ThreeBodyEquations, w, time_span, args=(results,))

    w[r1_i] += delta_r
    dev_three_body_sol = sci.integrate.odeint(ThreeBodyEquations, w, time_span, args=(results,))

    r1_sol = in_three_body_sol[:, :3]
    devr1_sol = dev_three_body_sol[:, :3]

    #lyapunov calculations Body 1
    dev_sol_r1 = np.absolute(np.linalg.norm(devr1_sol - r1_sol, axis=1))
    in_sol_r1 = np.absolute(np.linalg.norm(r1_sol, axis=1))
    results_r1 = np.log(dev_sol_r1 / in_sol_r1)

    #claculate slope
    slope, _, _, _, _ = stats.linregress(time_span, results_r1)

    print(f'Trial {this_trial} done')
    
    return slope


if __name__ == '__main__':
    with Pool() as p:
        outcome = p.map(main, range(trials))

    print("mean: ", np.mean(outcome))
    print("standard deviation: ", np.std(outcome))
