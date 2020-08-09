# Import numpy
import numpy as np
import scipy as sci

lamdas = []
for l in range(0,10000):
    # Define universal gravitation constant
    G = 6.67408e-11  # N-m2/kg2
    # Reference quantities
    m_nd = 1.989e+30  # kg #the solar mass (M☉)
    r_nd = 35.6 * 1.496e+11 # m #35.6 times the astronomical unit (AU)
    v_nd = 30000
    t_nd = 365 * 24 * 3600  # s #one year in seconds
    # Net constants
    K1 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
    K2 = v_nd * t_nd / r_nd

    # Define masses (solar mass)
    m1 = 1
    m2 = 1
    m3 = 1

    import random
    # Define initial position vectors
    r1 = [random.uniform(0.5,1), random.uniform(0.5,1), random.uniform(0.5,1)]
    r2 = [random.uniform(0.5,1), random.uniform(0.5,1), random.uniform(0.5,1)]
    r3 = [random.uniform(0.5,1), random.uniform(0.5,1), random.uniform(0.5,1)]
    # Convert pos vectors to arrays
    r1 = np.array(r1, dtype="float64")
    r2 = np.array(r2, dtype="float64")
    r3 = np.array(r3, dtype="float64")

    # Find Centre of Mass
    r_com=(m1*r1+m2*r2+m3*r3)/(m1+m2+m3)

    # Define initial velocities
    v1 = [0.01, 0.01, 0]  # m/s
    v2 = [-0.05, 0, -0.1]  # m/s
    v3 = [0, -0.01, 0] # m/s
    # Convert velocity vectors to arrays
    v1 = np.array(v1, dtype="float64")
    v2 = np.array(v2, dtype="float64")
    v3 = np.array(v3, dtype="float64")
    # Find velocity of COM
    v_com=(m1*v1+m2*v2+m3*v3)/(m1+m2+m3)

    #def calculate(devr1, devr2, devr3, w, t, G, m1, m2, m3):
    def ThreeBodyEquations(w, t, G, m1, m2, m3):
        r1 = w[:3]
        r2 = w[3:6]
        r3 = w[6:9]
        v1 = w[9:12]
        v2 = w[12:15]
        v3 = w[15:18]
        r12 = sci.linalg.norm(r2 - r1)
        r13 = sci.linalg.norm(r3 - r1)
        r23 = sci.linalg.norm(r3 - r2)

        dv1bydt = K1 * m2 * (r2 - r1) / r12 ** 3 + K1 * m3 * (r3 - r1) / r13 ** 3
        dv2bydt = K1 * m1 * (r1 - r2) / r12 ** 3 + K1 * m3 * (r3 - r2) / r23 ** 3
        dv3bydt = K1 * m1 * (r1 - r3) / r13 ** 3 + K1 * m2 * (r2 - r3) / r23 ** 3
        dr1bydt = K2 * v1
        dr2bydt = K2 * v2
        dr3bydt = K2 * v3
        r12_derivs = np.concatenate((dr1bydt, dr2bydt))
        r_derivs = np.concatenate((r12_derivs, dr3bydt))
        v12_derivs = np.concatenate((dv1bydt, dv2bydt))
        v_derivs = np.concatenate((v12_derivs, dv3bydt))
        derivs = np.concatenate((r_derivs, v_derivs))
        return derivs

    #Package initial parameters
    def initial(devr1, devr2, devr3):
        init_params=np.array([r1+devr1,r2+devr2,r3+devr3,v1,v2,v3]) #Initial parameters
        init_params=init_params.flatten() #Flatten to make 1D array
        return init_params

    # define time span
    years=400
    time_span=np.linspace(0,years,100*years) #20 orbital periods and 500 points

    delta_r = 1*(10**(-10))
    #Run the ODE solver
    import scipy.integrate
    in_three_body_sol = sci.integrate.odeint(ThreeBodyEquations,initial(0,0,0),time_span,args=(G,m1,m2,m3))
    dev_three_body_sol = sci.integrate.odeint(ThreeBodyEquations,initial(delta_r ,0,0),time_span,args=(G,m1,m2,m3))

    r1_sol = in_three_body_sol[:,:3]
    devr1_sol = dev_three_body_sol[:,:3]

    #lyapunov calculations Body 1
    results_r1=[]
    for i in range(0, len(time_span)):
        dev_sol_r1=np.absolute(np.linalg.norm(devr1_sol[i]-r1_sol[i]))
        in_sol_r1=np.absolute(np.linalg.norm(r1_sol[i]))
        diff=np.log(dev_sol_r1/in_sol_r1)
        results_r1.append(diff)

    from scipy import stats
    #claculate slope
    slope,_,_,_,_ = stats.linregress(time_span,results_r1)

    lamdas.append(slope)

print("mean: ", np.mean(lamdas))
print("standard deviation: ", np.std(lamdas))
