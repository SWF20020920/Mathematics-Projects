import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matrixSize = 100
sValue = 0.3
alphaValue = 1.31
iterSteps = 200

pattern = 1
isBilingual = 1

def languageDynamics(matrixSize, pattern, sValue, alphaValue, iterSteps, isBilingual):
    outputList = []
    fig = plt.figure()

    def setUp(dim = matrixSize, prob = 0.5, mode = pattern):
        if mode == 1:
            start = np.ones((dim, dim))
            start[20:-20, 20:-20] = 0
            return start, np.zeros((dim, dim))

        if mode == 2:
            start = np.ones((dim, dim))
            start[20:-20, 20:-20] = 0
            start[40:-40, 40:-40] = 1
            return start, np.zeros((dim, dim))

        if mode == 3:
            start = np.ones((dim, dim))
            start[22:30, 22:62] = 0
            start[30:38, 30:38] = 0
            start[30:38, 62:70] = 0
            start[38:46, 30:38] = 0
            start[38:46, 62:70] = 0
            start[46:54, 30:62] = 0
            start[54:62, 30:38] = 0
            start[54:62, 62:70] = 0
            start[62:70, 30:38] = 0
            start[62:70, 62:70] = 0
            start[70:78, 22:62] = 0
            return start, np.zeros((dim, dim))
        
        
        A = np.zeros((dim, dim))
        B = np.zeros((dim, dim))

        for i in range(0, dim):
            for j in range (0, dim):
                A[i][j] = np.random.binomial(1, prob)

        return A, B

    ### When mode = 1, we consider bilingual case. Otherwise, we use the paper's approach.
    def elementOutput(A, r, c, mode = isBilingual, s = sValue, alpha = alphaValue):
        value = A[r][c]
        zeroCounts, oneCounts, halfCounts = countNeighbors(A, r, c, mode)
        zeroCounts = zeroCounts + halfCounts
        oneCounts = oneCounts + halfCounts
        total = zeroCounts + oneCounts + halfCounts
        x = oneCounts / total
        xhat = max(min(0.67, x), 0.33)
        chance = np.random.binomial(1, 1 - 5*(0.5-xhat)**2)

        if ((mode == 1 and x > 0.33 and x < 0.67) or value == 0.5):
            if (chance == 1):
              return 0.5
            if (chance == 0):
              return np.random.binomial(1, 0.5)
        
        if (value == 0):
            prob = s * (x ** alpha)
            return np.random.binomial(1, prob)

        if (value == 1):
            prob = (1 - s) * ((1 - x) ** alpha)
            return np.random.binomial(1, 1 - prob)


    ### When mode = 1, we select diagonal neighbors as well. Otherwise, we use the paper's approach.
    def getNeighbors(A, r, c, mode = 1):
        size = int(np.sqrt(np.size(A)))

        if (r == 0):
            if(c == 0):
                return [(1, 0), (0, 1), (1, 1)]
            if(c == size - 1):
                return [(1, c), (0, c-1), (1, c-1)]
            else:
                return [(0, c-1), (0, c+1), (1, c-1), (1, c), (1, c+1)]
            
        if (r == size - 1):
            if(c == 0):
                return [(r, 1), (r-1, 0), (r-1, 1)]
            if(c == size - 1):
                return [(r, c-1), (r-1, c), (r-1, c-1)]
            else:
                return [(r, c-1), (r, c+1), (r-1, c-1), (r-1, c), (r-1, c+1)]
                
        if(c == 0):
            return [(r-1, 0), (r+1, 0), (r-1, 1), (r, 1), (r+1, 1)]
        if(c == size - 1):
            return [(r-1, c), (r+1, c), (r-1, c-1), (r, c-1), (r+1, c-1)]
        else:
            return [(r-1, c-1), (r-1, c), (r-1, c+1), (r, c-1), (r, c+1), (r+1, c-1), (r+1, c), (r+1, c+1)]

        if (mode == 0):
          if (r == 0):
              if(c == 0):
                  return [(1, 0), (0, 1)]
              if(c == size - 1):
                  return [(1, c), (0, c-1)]
              else:
                  return [(0, c-1), (0, c+1), (1, c)]
          
          if (r == size - 1):
              if(c == 0):
                  return [(r, 1), (r-1, 0)]
              if(c == size - 1):
                  return [(r, c-1), (r-1, c)]
              else:
                  return [(r, c-1), (r, c+1), (r-1, c)]
          
          if(c == 0):
              return [(r-1, 0), (r+1, 0), (r, 1)]
          if(c == size - 1):
              return [(r-1, c), (r, c-1), (r, c+1)]
          else:
              return [(r-1, c), (r, c-1), (r, c+1), (r+1, c)]

    def countNeighbors(A, r, c, mode):
        neighbors = getNeighbors(A, r, c, mode)
        zeroCounts = 0
        oneCounts = 0
        halfCounts = 0
        size = np.size(neighbors) / 2

        for i in range(int(size)):
            index = neighbors[i]
            element = A[index[0]][index[1]]
            if (element == 1):
                oneCounts += 1
            if (element == 0):
                zeroCounts += 1
            if (element == 0.5):
                halfCounts += 1
        return zeroCounts, oneCounts, halfCounts

    def updateMatrix(A, B):
        size = int(np.sqrt(np.size(A)))

        for i in range(size):
            for j in range(size):
                B[i][j] = elementOutput(A, i, j)

        return A, B

    fig = plt.figure()

    def iteration(num = iterSteps):
        A, B = setUp()
        for i in range(num):
          B, A = updateMatrix(A, B)
          print("Current iteration: ", i + 1)
          if (i % 2 == 1):
            p = plt.matshow(A, cmap=plt.cm.get_cmap('Blues', 6))
            plt.colorbar()
            plt.clim(0, 1)
            plt.show()
            outputList.append(p)
            plt.savefig('my_plot.png')
          if (i % 2 == 0):
            p = plt.matshow(B, cmap=plt.cm.get_cmap('Blues', 6))
            plt.colorbar()
            plt.clim(0, 1)
            plt.show()
            outputList.append(p)
            plt.savefig('my_plot.png')
        return A, B

    iteration()

    def buildanim(i=int):
        p = outputList[i]
        return p
    

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, buildanim, interval = 2)
    plt.show()

languageDynamics(matrixSize, pattern, sValue, alphaValue, iterSteps, isBilingual)
