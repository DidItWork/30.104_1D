import matplotlib.pyplot as plt
import numpy as np
import displacement as disp
from scipy.integrate import odeint
from scipy.optimize import minimize

class simulation:

    def __init__(self, d, T, a, equation=disp.polynomial):
        self.d = d
        self.T = T
        self.a = a
        self.k = 1
        self.m_b = 1
        self.b = 0.2
        self.penalty_weight = 10
        self.penalty_weight2 = 10
        self.res = T/100 #Simulation Resolution

        self.x = np.array([])
        self.t = np.array([])
        self.equation = equation

    def get_d(self):
        
        return self.d
    
    def get_b(self):

        return self.b

    def get_T(self):

        return self.T
    
    def get_a(self):

        return self.a
    
    def get_k(self):
        
        return self.k
    

        return self.L_0
    
    def get_m_b(self):
        
        return self.m_b
    
    def get_res(self):

        return self.res

    def set_d(self, d):
        
        self.d = d

    def set_b(self, b):
        
        self.b = b

    def set_T(self, T):

        self.T = T
    
    def set_a(self, a):

        self.a = a
    
    def set_k(self, k):
        
        self.k = k
    
    def set_m_b(self, m_b):
        
        self.m_b = m_b

    def set_res(self, res):

        self.res = res

    def x_c(self, d, t, a, T):
        return self.equation(d, t, a, T)
        # return displacement.cosine_harmonics(d, t, a, T)

    def x_c_dot(self, t):

        return self.equation(self.d,t,self.a, self.T)
    
    def x_c_dotdot(self, t):

        return self.equation(self.d,t,self.a, self.T)

    def f(self, u, t):

        # return (u[1],(-self.b*(self.x_c_dot(t)+u[1])-self.k*(u[0]-self.L_0)-self.m_b*self.x_c_dotdot(t))/self.m_b)
        return (u[1],(-self.b*u[1]+self.k*(self.x_c(self.d, t, self.a, self.T)-u[0]))/self.m_b)

    def linear_penalty(self, penalty_weight, t):
        """Linear penalty function for negative values"""
        result = self.equation(self.d, t, self.a, self.T)
        # setting penalty weight to be 1, can change penalty weight
        return sum([(max(0, -x)+max(0,x-self.d))**2 for x in result]) * penalty_weight * (t[1]-t[0])

    def reverse_penalty(self, penalty_weight2, t):
        #Penalty to make sure curve is strictly increasing
        result = self.equation(self.d, t, self.a, self.T)
        penalty = 0
        for i in range(1, len(result)):
            if result[i] < result[i-1]:
                penalty += (result[i-1]-result[i])**2
        return penalty * penalty_weight2 * (t[1]-t[0])


    def cost(self, a):

        #args = [T, a1, a2, ...]
        res = self.res

        self.set_a(a)

        t = np.arange(0,self.T*5,res)

        c_0 = [0,0] #values for x and x_dot respectively at t=0

        result = odeint(self.f, c_0, t)

        x = np.array(result[:,0])

        x_ss = result[len(x)-1,0]

        self.x = x.copy()
        self.t = t.copy()

        x -= self.d

        # print(x)

        t_i = 0

        for i in range(x.shape[0]):
            if x[i] > 0 or t[i]>self.T:
                t_i = i
                break

        c_v = np.concatenate((np.zeros(t_i+1),np.ones(x.shape[0]-t_i-1)))

        cost = np.matmul(c_v,x**2)+self.reverse_penalty(self.penalty_weight2, t)

        # print(cost)
        
        return cost

    def get_cost(self):
        return self.cost(self.a)

    def plot_result(self):
        plt.plot(self.t,self.x)
        plt.title("Displacement of print bed x/m against t/s")
        plt.xlabel("Time t/s")
        plt.ylabel("Displacement of bed x/m")
        plt.show()

    def optimize(self):

        #optimizing a

        print("Optimizing...")

        res = minimize(self.cost, self.a, method='nelder-mead', options={'xatol':1e-8,'disp':True})

        self.a = res.x

        print("res.x:",res.x)


def main():

    #Simulation body

    #Model Parameters
    #a = np.array([3.88144156, -8.7827797,  -1.69893668, 15.46472629])
    # a = np.array([6.41890472, -16.98920625,  14.74610913,  10.29036702])
    # a = np.array([1.25154432e+11, -2.85699456e+11,  1.78183933e+11, -1.02775459e+10])
    # a = np.array([23.37247036, -31.51913279, -11.69200124,  22.59164243])
    a_array = np.array([[7.57019358e-03, -8.05906487e-05,  5.97101546e-04, -2.42414153e-04]])
    a = np.array([7.57225490e-03, -8.06146837e-05,  5.97337219e-04, -2.42577751e-04])
    d = 10
    T = 10
    print("printing displacement cosine_harmonics")
    # displacement.plot_cosine_harmonics(d, t,displacement.cosine_harmonics, a, T)
    #displacement.plot_cosine_power(d, t,displacement.cosine_power, a, T)
    disp.plot(d, disp.cosine_harmonics, T, a_array)
    sim = simulation(d, T, a, disp.cosine_harmonics)
    sim.set_k(1)
    cost = sim.get_cost()
    print("cost:",cost)
    sim.plot_result()
    sim.optimize()
    sim.plot_result()

    # a1, a2, a3
    # sim.cost(T, a1, a2, a3)

    

if __name__ == "__main__":
    
    #Running main program
    main()
