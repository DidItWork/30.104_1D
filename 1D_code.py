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

    def solve(self,a):

        self.set_a(a)

        t = np.arange(0,self.T*5,self.res)

        self.t = t.copy()

        c_0 = [0,0] #values for x and x_dot respectively at t=0

        result = odeint(self.f, c_0, t)

        return np.array(result[:,0])


    def cost(self, a):

        x = self.solve(a)

        self.x = x.copy()

        x -= self.d

        # print(x)

        t_i = 0

        c_v = np.array([])

        for i in range(x.shape[0]):
            if self.t[i]>self.T:
                t_i = i
                break
            if x[i] > 0 or x[i]<-self.d:
                c_v = np.concatenate((c_v,np.ones(1)))
            else:
                c_v = np.concatenate((c_v,np.zeros(1)))
        
        c_v = np.concatenate((c_v,np.ones(x.shape[0]-t_i)))

        cost = np.matmul(c_v,x**2)+self.reverse_penalty(self.penalty_weight2, self.t)

        return cost

    def get_cost(self):
        return self.cost(self.a)

    def plot_result(self):
        plt.plot(self.t,self.x)
        plt.title("Displacement of print bed x against t")
        plt.xlabel("Time t")
        plt.ylabel("Displacement of bed x")
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
    # a_array = np.array([[1,1,1,1,1],[3.10882612, -0.83610475, -4.45310798,  6.09282191, -2.16597904]]) #power
    a_array = np.array([[1,1,1,1,1],[2.35360779, -3.01336176, -3.79217689,  6.42038199, -1.88190296]]) #harmonics
    # a = np.array([7.57225490e-03, -8.06146837e-05,  5.97337219e-04, -2.42577751e-04])
    # a = np.array([ 11.42569133, -15.95800003, -11.58504045,  24.25121788,  -7.29509888])
    a = np.array([1,1,1,1,1,1])
    d = 10
    T = 10
    print("printing displacement cosine_harmonics")
    # disp.plot_cosine_harmonics(d, disp.cosine_harmonics, a, T)
    #disp.plot(d, disp.cosine_power, T, np.array([a]))
    # disp.plot(d, disp.cosine_harmonics, T, a_array)
    sim = simulation(d, T, a, disp.cosine_power)
    args_list = [[10,0.1,10,10,2],[10,0.1,10,5,2]]
    for args in args_list:
        sim.set_k(args[0])
        sim.set_b(args[1])
        sim.set_d(args[2])
        sim.set_T(args[3])
        sim.set_m_b(args[4])
        # x = sim.solve(a_array[0])
        # x_= sim.solve(a_array[1])
        # t = sim.t
        # plt.plot(t,x,label="Unoptimized")
        # plt.plot(t,x_,label="Optimized")
        # plt.legend()
        # plt.title("Displacement of print bed x against t")
        # plt.xlabel("Time t")
        # plt.ylabel("Displacement of bed x")
        # plt.show()
        cost = sim.get_cost()
        print("cost:",cost)
        sim.plot_result()
    sim.optimize()
    sim.plot_result()

    # a1, a2, a3
    # sim.cost(T, a1, a2, a3)

def plot_results():
    x = np.array([0,1])
    m_b = np.array([946.15,1709.9])
    d_T = np.array([365.82,2290.2])
    b = np.array([1735.8,920.25])
    k = np.array([1740.8,915.24])

    plt.plot(x,m_b,label="m_b")
    plt.plot(x,d_T,label="d/T")
    plt.plot(x,b,label="b")
    plt.plot(x,k,label="k")
    plt.legend()
    plt.title("Effect of doubling parameters on Cost")
    plt.ylabel("Average Cost C")
    plt.xlabel("Value of Parameter")
    plt.show()

if __name__ == "__main__":
    
    #Running main program
    main()
    # plot_results()
