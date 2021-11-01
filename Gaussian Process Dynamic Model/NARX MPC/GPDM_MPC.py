"""
XXX

S.P. Mulders (Sebastiaan)
Delft Center for Systems and Control (DCSC)
The Netherlands, 2021
"""
import GPy
import casadi as csd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore") # To suppress numpy error (checked that the error can be ignored)

# Set a fixed seed for demo purposes
np.random.seed(seed=1)

#%% True system model
def f(x,u,d,p,h):
    k1  = F(x,u,d,p,h)
    k2  = F(x + h/2 * k1,u,d,p,h)
    k3  = F(x + h/2 * k2,u,d,p,h)
    k4  = F(x + h * k3,u,d,p,h)
    x   = x + h/6*(k1 + 2*k2 + 2*k3 + k4)
    dx  = x
    return dx

def F(x,u,d,p,h):
    wn   = 0.8;
    zeta = .3;
    
    A    = np.array([[0, 1], [-wn**2,-2*zeta*wn]])
    B    = np.array([[0], [wn**2]])
    # Bd   = np.ones((B.size, 1))
    ki   =  np.dot(A, x) + np.dot(B, u) + np.dot(B, d)
    return ki

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
         return tempTimeInterval

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def xt_func(ut):
    return np.nan_to_num(xt, copy=True, nan=ut)

def information_gain(m, ut):
    xt_tmp = xt_func(ut)
    _, sigma2 = m.predict(xt_tmp[np.newaxis])
    
    return sigma2

def randomBoundedInput(u_min, u_max):
    return np.random.uniform(low=u_min, high=u_max)

def generateReferenceSignal():
    u = np.ones(N+Np)
    u[0:50] = 0.5
    u[50:150] = -0.25
    u[150:250] = 0.0
    u[250:350] = 1.0
    u[350:N+Np] = -0.75
    
    return u

def euclideanDistanceR(Xnew, X, lengthscale, forCasadi=False):
    X2 = np.sum(np.square(X),1)
    if forCasadi:
        Xnew2 = csd.sum1(Xnew**2)
        r2 = -2.*csd.mtimes(Xnew.T, X.T) + (Xnew2 + X2[np.newaxis])
        r = csd.sqrt(r2)
    else:
        Xnew2 = np.sum(Xnew**2)
        r2 = -2.*np.dot(Xnew.T, X.T) + (Xnew2 + X2[np.newaxis])
        r = np.sqrt(r2)
       
    return r/lengthscale

def K_of_r(r, variance):
    K = variance*csd.exp(-0.5 * r**2)
    return K
    
def gpPredict(Xnew, X, lengthscale, variance, woodbury_vector, forCasadi=False):
    r = euclideanDistanceR(Xnew, X, lengthscale, forCasadi)
    Kx = K_of_r(r, variance)
    mu = csd.mtimes(Kx, woodbury_vector)
    
    return mu

#%% Learning the dynamic GP model 

Ts = 0.1;
T = 50
t_arr = np.arange(0, T, Ts)

runMode = 2 # 1 = perform GP optimization, 2 = perform simulations with GP model and true system
makePlots = 1 # For runMode = 1: Shows the variance of the prediction; For runMode = 2: applies a random input signal and plots the true system and trained GP model.
enableMPC = 1 # 0 = Only make plots of the open-loop system with a predefined reference input u, 1 = Enables MPC to 'optimally' track the reference r [only used when runMode = 2]
varianceDisturbance = 0 # Scale factor for the Gaussian white noise disturbance.
includeDisturbances = 0 # Whether to include the disturbance in the regression vector x_t [NOT YET IMPLEMENTED!]
enableOptimalExperimentDesign = 0 # 0 = apply randomBoundedInput, 1 = WIP, Optimal Experiment Design [only used when runMode = 1]
kernelType = 'RBF' # Possible options: 'RBS', 'Matern32', 'Matern52', 'Exponential' [only used when runMode = 1]
plotInterval = 5 if enableMPC else 10

p = 0
ix_y = 0 # Output state

Np = 15 # MPC prediction horizon [samples][only used when runMode = 2]
N = np.floor(T/Ts).astype(int)-Np # Number of samples to simulate [only used when runMode = 2]
Q, R, lam = 1.0, 0.1, 1.0 # Penalization matrices on output and input

k_arr = [40] # Amount of lagged terms y/u/(w) to include in the regression vector x_t

#%% 
if makePlots:
    fig, ax = plt.subplots()

for k in k_arr:
    # Past lags for output, input and disturbance signals 
    # \dim{y} = k-1, \dim{u} = k, \dim{w} = k
    # k = 40
    kdim = (2+includeDisturbances)*k-1
    xt = np.zeros(kdim)
    
    x = np.array([[0],[0]])
    ut = 0
    y_prev = 0
    
    # Actuation constraints
    u_max = 1 # Saturation maximum
    u_min = -1 # Saturation minimum
    u_dot = 1 # Allowed rate of change per second
    
    # Pre-allocate arrays to store data
    if runMode == 1:
        x_arr = np.zeros((2, len(t_arr)))
    elif runMode == 2:
        x_arr = np.zeros((2, N))
    u_arr = np.zeros((len(t_arr),1))
    y_arr = np.zeros((len(t_arr),1))
    y_gp_mu_arr = np.zeros((len(t_arr),1))
    xt_arr = np.zeros(((2+includeDisturbances)*len(t_arr),kdim))
    sigma2_arr = np.zeros((len(t_arr),1))
    sigma2_tmp_arr = np.zeros((len(t_arr),101))
    
    yt_arr = np.zeros(k-1)
    ut_arr = np.zeros(k)
    wt_arr = np.zeros(k)
    
    if runMode == 1:
        # Initialize GP
        if kernelType == 'RBF':
            kernel = GPy.kern.RBF(input_dim=kdim, variance=1., lengthscale=1.)
        elif kernelType == 'Matern32':
            kernel = GPy.kern.Matern32(input_dim=kdim, variance=1., lengthscale=1.)
        elif kernelType == 'Matern52':
            kernel = GPy.kern.Matern52(input_dim=kdim, variance=1., lengthscale=1.)
        elif kernelType == 'Exponential':
            kernel = GPy.kern.Exponential(input_dim=kdim, variance=1., lengthscale=1.)
        else:
            raise Exception('Incorrect kernelType specified')
            
        #%% Simulate and learn GP
        for ii, t in enumerate(t_arr):
            # If this is the very first iteration, simulate 
            # if ii == 1:
            # for jj in range(k)
            w = np.random.normal(loc=0, scale=varianceDisturbance)
            
            ut_arr = np.append(ut_arr[1:], np.nan)
            yt_arr = np.append(yt_arr[1:], y_prev)
            
            xt[:k-1] = yt_arr
            xt[k-1:] = ut_arr
            
            #### Optimal Experiment Design
            
            if ii > k:
                sigma2_tmp = np.array([])
                if enableOptimalExperimentDesign:
                    ut_tmp_arr = np.linspace(u_min, u_max, 101)
                    
                    for ut_tmp in ut_tmp_arr:
                        sigma2_tmp = np.append(sigma2_tmp, information_gain(m, ut_tmp))
                    
                    sigma2_argmax = np.argmax(sigma2_tmp)
                    sigma2_arr[ii-(k+1)] = sigma2_tmp[sigma2_argmax]
                    sigma2_tmp_arr[ii-(k+1)] = sigma2_tmp
                    
                    
                    print('Maximum variance was found at index #{loc}'.format(loc=sigma2_argmax))
                    
                    ut = ut_tmp_arr[sigma2_argmax]
                else:
                    ut = randomBoundedInput(u_min, u_max)
                    sigma2_tmp = np.append(sigma2_tmp, information_gain(m, ut))
                    sigma2_arr[ii-(k+1)] = sigma2_tmp
                
                if makePlots:
                    ax.clear()
                    ax.semilogy(t_arr[k:ii], sigma2_arr[:ii-k])
                    ax.grid()
                    ax.set_xlim([0, T])
                    ax.set_title('Variance output prediction')
                    ax.set_xlabel('Time [s]')
                    ax.set_ylabel('Variance')
                    plt.pause(0.00001)
            else:
                ut = randomBoundedInput(u_min, u_max)
            
            u_arr[ii][0] = ut
            np.nan_to_num(ut_arr, copy=False, nan=ut)
            
            #### /Optimal Experiment Design
            
            xt[:k-1] = yt_arr
            xt[k-1:] = ut_arr
            
            xt_arr[ii] = xt
        
            # Apply ut to system and meaure x
            x = f(x, ut, w, p, Ts)
            x_arr[:, ii] = x.squeeze()
            
            yt = x[ix_y][0]    
            y_arr[ii][0] = yt
            
            if ii >= k:
                X = xt_arr[:ii+1]
                Y = y_arr[:ii+1]
                
                if ii == k:
                    m = GPy.models.GPRegression(X, Y, kernel)
                    m.optimize('bfgs')
                else:
                    m.set_XY(X=X, Y=Y)
                    m.optimize('bfgs')
            
            y_prev = yt
            
            # Print status message
            if ii%10 == 0:
                print('Progress k = {k}: {:2.2%}, this step took {:.1f} seconds'.format(ii/len(t_arr), toc(), k=k))
                tic()
        
        fileSaveName = 'GP_EX2_K{k}_W{w}_T{T}'.format(w=str(float(varianceDisturbance)).replace('.',''), k=k, T=T)
        m.save_model(fileSaveName)
        print('Model was saved with file name: ' + fileSaveName)
    elif runMode == 2:
        fileLoadName = 'GP_EX2_K{k}_W{w}_T{T}.zip'.format(w=str(float(varianceDisturbance)).replace('.',''), k=k, T=T)
        m = GPy.Model.load_model(fileLoadName)
        x = np.array([[0],[0]])
        
        ref = generateReferenceSignal()
          
        y_gp_mu_arr = np.zeros(N)
        y_gp_var_arr = np.zeros(N)
        
        y_prev = x[ix_y][0]
        y_gp_mu_prev = x[ix_y][0]
        y_gp_var = 0
        
        for ii in range(N):
            w = 0 #np.random.normal(loc=0, scale=varianceDisturbance)
            x_arr[:, ii] = x.squeeze()
            
            if enableMPC:      
                opti = csd.Opti();
                u_opt = opti.variable(Np)
                y_opt = opti.variable(Np)
                sigma2_opt = opti.variable(Np)
              
                rf = ref[ii:ii+Np]
              
                ut_opt_arr = csd.MX(np.append(ut_arr, np.zeros(Np)))
                yt_opt_arr = csd.MX(np.append(yt_arr, np.zeros(Np)))
                xt_opt = csd.MX(np.zeros(kdim))
              
                opti.minimize( csd.mtimes(csd.mtimes((rf-y_opt).T, Q*np.eye(Np)), (rf-y_opt)) )#+ \
                               # csd.mtimes(csd.mtimes(u_opt.T, R*np.eye(Np)), u_opt) ) # + \
                               # csd.mtimes(csd.mtimes(sigma2_opt.T, lam*np.eye(Np)), sigma2_opt) )
                for jj in range(Np):
                    ut_opt_arr[k+jj] = u_opt[jj]
                    if jj > 0:
                        yt_opt_arr[k-1+jj] = y_opt[jj-1]
                                      
                    xt_opt[:k-1] = yt_opt_arr[jj:k-1+jj]
                    xt_opt[k-1:] = ut_opt_arr[1+jj:k+jj+1]
                  
                    mu = gpPredict(xt_opt, m.X, m.kern.lengthscale[0], m.kern.variance[0], m.posterior.woodbury_vector, forCasadi=True)
                  
                    opti.subject_to( y_opt[jj] == mu )
                    opti.subject_to( u_opt[jj] >= u_min )
                    opti.subject_to( u_opt[jj] <= u_max )
                    # if jj == 0:
                        # opti.subject_to( csd.fabs(u_opt[jj] - ut_arr[-1]) <= 10*u_dot*Ts )
                    # else:
                        # opti.subject_to( csd.fabs(u_opt[jj] - u_opt[jj-1]) <= 10*u_dot*Ts )
    
                p_opts = {} # {"expand":True}
                s_opts = {'print_level': 5, "max_iter": 1000}    
                opti.solver('ipopt', p_opts, s_opts)
                sol = opti.solve()
              
                u = sol.value(u_opt)[0]
                
                ut_arr = np.append(ut_arr[1:], u)
                yt_arr = np.append(yt_arr[1:], y_prev)
                
                xt[k-1:] = ut_arr
                xt[:k-1] = yt_arr
                xt_arr[ii,:] = xt
                
            else:
                y_gp_mu_arr[ii] = y_gp_mu_prev
                y_gp_var_arr[ii] = y_gp_var
            
                u = ref[ii]
              
                ut_arr = np.append(ut_arr[1:], u)
                yt_arr = np.append(yt_arr[1:], y_gp_mu_prev)
              
                xt[:k-1] = yt_arr
                xt[k-1:] = ut_arr
                xt_arr[ii,:] = xt
              
                y_gp_mu, y_gp_var = m.predict(xt[np.newaxis], include_likelihood=False)
                y_gp_mu_prev = y_gp_mu[0][0]
              
            x = f(x, u, w, p, Ts)    
            y = x[ix_y][0]
            
            y_arr[ii] = y
            u_arr[ii] = u
            
            y_prev = y
            
            if ii%plotInterval == 0:
                print('Time {:.2f}, u={:.2f}'.format(ii*Ts, u))
                ax.clear()
                ax.plot(t_arr[:N], ref[:N], '--', label='Reference', linewidth=1.0, color='0.5')
                ax.plot(t_arr[:ii+1], u_arr[:ii+1], label='Input', linewidth=1, color='0')
                ax.plot(t_arr[:ii+1], y_arr[:ii+1], 'g', label='Output - True system', linewidth=2)
                if not enableMPC:
                    ax.plot(t_arr[:ii+1], y_gp_mu_arr[:ii+1], '-.k', label='Output - GP', linewidth=1)
                ax.grid()
                ax.set_xlim([0, (N-1)*Ts])
                ax.set_title('GP-MPC reference tracking progress')
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Output')
              
                ax.legend(loc=1)
                plt.pause(0.01)
    else:
        raise Exception('Invalid runMode')

#%% Verification and plots
# an = 0.001;

# t = np.arange(0, 1200, Ts);
# wn = an*t;
# u = np.sin(0.5*wn*t);

# x = np.array([[0], [0]])
# x_array = np.zeros((2, len(t)))
# for ii, uk in enumerate(u):
#     d = np.random.normal(loc=0.0,scale=0.1);
#     p = 0;
#     x = f(x, uk, d, p, Ts);
#     x_array[:, ii] = x.squeeze();

# plt.plot(wn, u)
# plt.plot(wn, x_array[0,:])
# plt.plot(wn, x_array[1,:])