import numpy as np
import matplotlib.pyplot as plt

def cosine_harmonics(d,t,param_array,T):
    t = np.clip(t,0,T)  #for values <0, set to 0 and for values >T, set to T
    n = len(param_array)
    den = np.sum(param_array)
    num = 0
    for i in range(0,n):
        num += (param_array[i]/2)*(1-np.cos((2*i+1)*np.pi*(t/T)))
    e = d*num/den
    return e
    
def cosine_power(d,t,param_array,T):
    t = np.clip(t,0,T)  #for values <0, set to 0 and for values >T, set to T
    n = len(param_array)
    den = np.sum(param_array)
    num = 0
    for i in range(3,n+3):
        num +=param_array[i-3]*(1-(np.cos(((np.pi*t)/(2*T))))**i)
    e = d*num/den
    return e

def polynomial(d,t,param_array,T):
    t = np.clip(t,0,T)  #for values <0, set to 0 and for values >T, set to T
    n = len(param_array)
    den = 0
    num = np.zeros_like(t)
    powers = np.array(list(range(3,n+3)))
    for i in range(3,n+3):
        num += param_array[i-3]*(((-t**i)/i)+(T*(t**(i-1)))/(i-1))
        den += param_array[i-3]*((T**i)/(i*(i-1)))
    e = d*num/den
    return e

def plot(d, equation, T, param_arrays):
    t = np.linspace(0,T,200)
    for i in range(param_arrays.shape[0]):
        plt.plot(t,equation(d,t,param_arrays[i],T),label = "n = "+str(len(param_arrays[i])))
    plt.xlabel("t")
    plt.ylabel("x_c")
    plt.title("x_c against t")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    
    #f = np.zeros_like(t) # turn t array into 0
    T = 10
    d = 10
    t = np.linspace(0, T, 200) # array of t starting from 0 to T
    a = np.array([1.25154432e+11, -2.85699456e+11,  1.78183933e+11, -1.02775459e+10])
    penalty_weight = 1
    penalty_weight2 = 5
    # create an array for the parameters
    n = 4
    param_array = np.array(range(n,0,-1))
    #param_array = np.random.rand(n,1)

    # print("plotting displacement")
    # plot_cosine_harmonics(d,t,cosine_harmonics,param_array,T)

    # print(linear_penalty(penalty_weight,d,t,T,a))

    harmonics_args = np.array([3.52455935, -0.03786313,  0.30414784, -0.22359681,  0.08369796])
    power_args = np.array([1.81245345, -0.52092429, 3.37432895, -6.80384119, 3.79277465])

    plt.plot(t,cosine_harmonics(d,t,harmonics_args,T),label="Cosine Harmonics")
    plt.plot(t,cosine_power(d,t,power_args,T),label="Cosine Power")
    plt.xlabel("t")
    plt.ylabel("x_c")
    plt.title("x_c against t")
    plt.legend()
    plt.show()
