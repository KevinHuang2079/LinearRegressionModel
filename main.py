import numpy as np

#mean square Error
def computeError(b, m, points):
    totalError = 0 
    #sum all distances between the two y values and square it
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y - (m * x + b)) **2

    #return avg
    return totalError / float(len(points))

def gradientDescent(Points, starting_B, starting_M, learning_Rate, Epochs):
    b = starting_B
    m = starting_M
    for i in range(Epochs):
        b,m = gradientStep(b, m, np.array(Points),learning_Rate)
    return b,m

def gradientStep(currentB, currentM, Points, learning_Rate):
    bGradient = 0 
    mGradient = 0
    
    N = float(len(Points))

    for i in range (0,len(Points)):
        x = Points[i,0]
        y = Points[i,1]
        #partial derivatives with respect to m and b
        bGradient += -(2/N) * (y - ((currentM * x) + currentB))
        mGradient += -(2/N) * x * (y - ((currentM * x) + currentB))
        
        

    newB = currentB - (learning_Rate * bGradient)
    newM = currentM - (learning_Rate * mGradient)

    return [newB, newM]


def main():
    #load data from csv file
    points = np.genfromtxt('grades.csv', delimiter =',')

    learningRate = .0001
    startingB = 0
    startingM = 0
    numberIterations = 2000

    print(f"starting gradient descent at b = {startingB}, m = {startingM}, error = {computeError(startingB, startingM, points)}")
    [b, m] = gradientDescent(points, startingB, startingM, learningRate, numberIterations)
    print(f"ending point at b = {b}, m = {m}, error = {computeError(b, m, points)}")



if __name__ == '__main__':
    main()
