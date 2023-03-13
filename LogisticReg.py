import numpy as np

def linearf(x, w, b):
    z = np.dot(x, w) + b
    return z

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def compute_cost(x, y, w, b, lambda_=1):
    m = x.shape[0]
    
    z = linearf(x, w, b)
    f_wb = sigmoid(z)
    
    cost = -np.sum(y*np.log(f_wb) + (1-y)*np.log(1-f_wb))/m
    
    regcost = lambda_/(2*m) * np.sum(w**2)
    
    cost += regcost
    
    return cost

def compute_gradient(x, y, w, b, lambda_=1):
    m = x.shape[0]
          
    z = linearf(x, w, b)
    f_wb = sigmoid(z)
    
    dj_z = f_wb - y
    dj_w = np.dot(x.T, dj_z)/m
    dj_b = np.sum(dj_z)/m
    
    dj_w += lambda_/m * w
    
    return dj_w, dj_b
    
def batch_gradient_descent(x, y, learning_rate=0.1, num_iterations=15000, lambda_=1):
    m, n = x.shape
    w = np.zeros(n)
    b = 0
    
    for i in range(num_iterations):
        dj_w, dj_b = compute_gradient(x, y, w, b, lambda_)
        w -= learning_rate * dj_w
        b -= learning_rate * dj_b
        
    return w, b

def predict(x, w, b):
    z = linearf(x, w, b)
    y_pred = sigmoid(z)
    y_pred = (y_pred >= 0.5).astype(int)
    return y_pred

def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred)/len(y_true)
    return acc
