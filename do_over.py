from PIL import Image
import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import random
from random import shuffle
import copy
import time
from operator import itemgetter

imgname = 'panda'

imgArr = np.asarray(Image.open(imgname + '.jpg'), dtype=np.uint8)

# If not a 2D grayscale image, convert to grayscale
if len(imgArr.shape) != 2:
    imgArr = imgArr[:,:,0]

WIDTH, HEIGHT = imgArr.shape
USE_CIRCLES = True

def make_circle(x, y, r, c, a):
    return plt.Circle((x, -y), r, color='black', alpha=a)

def make_square(x, y, width, height, c, a):
    return plt.Rectangle((x, -y), width, height, color='black', alpha=a)

def squareFitness(square):
    x = square.get_x()
    y = -square.get_y()
    w = square.get_width()
    h = square.get_height()
    o = int(255 - square.get_alpha() * 255)

    diff = 0

    for i in range(int(x), int(x+w)):
        for j in range(int(y), int(y+h)):
            if j >= 0 and j < WIDTH and i >= 0 and i < WIDTH:
                diff += o - imgArr[(WIDTH-1)-i][j]
            else:
                diff += 255
    diff = np.abs(diff) / (w*h)
    return 1 - diff / 255

def circleFitness(cir):
    x = int(cir.center[0])
    y = int(-cir.center[1])
    r = int(cir.get_radius())
    o = int(cir.get_alpha() * 255)
    diff = 0
    for i in range(y-r, y+r):
        j = x
        while (j-x)**2 + (i-y)**2 < r**2:
            if j >= 0 and j < WIDTH and i >= 0 and i < WIDTH:
                diff += o - imgArr[(WIDTH-1)-i][j]
            else:
                diff += 255
            j -= 1
        j = x + 1
        while (j-x)*(j-x) + (i-y)*(i-y) < r*r:
            if j >= 0 and j < WIDTH and i >= 0 and i < WIDTH:
                diff += o - imgArr[(WIDTH-1)-i][j]
            else:
                diff += 255
            j += 1
    diff = np.abs(diff) / (np.pi * r * r)
    return 1-diff / 255

def fitness(genome):
    #print(genome.shape)
    #print(imgArr.shape)
    if genome.shape == imgArr.shape:
        diff = genome - imgArr
        return 1 - (np.absolute(np.sum(diff)) / (WIDTH * HEIGHT * 250))

    return -1

def graph_image(circles):
    ax.clear()
    for c in circles:
        newPatch = copy.copy(c)
        ax.add_patch(newPatch)
    plt.xlim(0, WIDTH)
    plt.ylim(-HEIGHT, 0)
    ax.invert_yaxis()
    plt.axis('off')
    plt.show()
    fig.canvas.draw()
    plt.get_current_fig_manager().resize(WIDTH,WIDTH+34)
    plt.pause(.001)
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data[:,:,0]
    return data

def init_pop(popSize, indvSize):
    population = []

    for p in range(popSize):
        indv = []
        for i in range(indvSize):
            x, y = random.randint(5, WIDTH-5), random.randint(5, HEIGHT-5)
            if USE_CIRCLES:
                radius = random.randint(2, 4)
                a = np.random.rand()/4
                indv.append(make_circle(x, y, radius, 'k', a))
            else:
                width = random.randint(2, 6)
                height = random.randint(2, 6)
                a = np.random.rand()
                indv.append(make_square(x, y, width, height, 'k', a))

        population.append(indv)

    return population

def bound(value, adjustment, mini, maxi):

    movement = value + adjustment
    if value + adjustment > maxi:
        movement = maxi - ((value + adjustment) - maxi)
    elif value + adjustment < mini:
        movement = mini + (mini - (value + adjustment))
    
    if movement < mini or movement > maxi:
        return bound(movement, 0, mini, maxi)

    return movement

fitCount = 0
def mutate(indvidual, generationNum, printFit):
    global fitCount
    fitCount = 0
    maxMovementChange = WIDTH
    maxHeightChange = 1
    maxWidthChange = 1
    maxRadiusChange = 4
    maxOpacityChange = .1
    minAlpha = 0.05
    minRadius = 2
    maxRadius = 4
    maxWidth = 8
    maxHeight = 8
    minWidth = 4
    minHeight = 4

    newCircles = []
    for i in range(len(indv)):
        fitness = 0
        if USE_CIRCLES:
            fitness = circleFitness(indv[i])
        else:
            fitness = squareFitness(indv[i])

        curviness = 1.1
        crossover = .99
        maxMutationAmount = (fitness*curviness - crossover*curviness)/(fitness - crossover*curviness)
        #if maxMutationAmount < 0: maxMutationAmount = 0
        if fitness >= crossover: 
	    maxMutationAmount = 0
	    fitCount += 1

        #print(fitness)
        #print(maxMutationAmount)
        #print("----")
        xMutateDirection = 1 if np.random.rand() > 0.5 else -1
        yMutateDirection = 1 if np.random.rand() > 0.5 else -1
        alphaMutateDirection = 1 if np.random.rand() > 0.5 else -1
        widthMutateDirection = 1 if np.random.rand() > 0.5 else -1
        heightMutateDirection = 1 if np.random.rand() > 0.5 else -1
        
        radiusMutateDirection = 1 if np.random.rand() > 0.5 else -1

        radiusMove =  np.random.rand() * maxRadiusChange * maxMutationAmount * radiusMutateDirection
        xMove = maxMovementChange * np.random.rand() * maxMutationAmount * xMutateDirection
        yMove = maxMovementChange * np.random.rand() * maxMutationAmount * yMutateDirection
        alphaMove = maxOpacityChange * np.random.rand() * maxMutationAmount * alphaMutateDirection
        widthMove = maxWidthChange * np.random.rand() * maxMutationAmount * widthMutateDirection
        heightMove = maxHeightChange * np.random.rand() * maxMutationAmount * heightMutateDirection
        
        if USE_CIRCLES:
            radius = bound(indv[i].get_radius(), radiusMove, minRadius, maxRadius)
            centerX = bound(indv[i].center[0], xMove, radius, WIDTH - radius)
            centerY = bound(indv[i].center[1], yMove, -HEIGHT + radius, -radius)
            alpha = bound(indv[i].get_alpha(), alphaMove, minAlpha, 1)
            newCircles.append(plt.Circle((centerX, centerY), radius, color=indv[i].get_edgecolor(), alpha=alpha))
        else:
            width = bound(indv[i].get_width(), widthMove, minWidth, maxWidth)
            height = bound(indv[i].get_height(), heightMove, minHeight, maxHeight)
            centerX = bound(indv[i].get_x(), xMove, width, WIDTH - width)
            centerY = bound(indv[i].get_y(), yMove, -HEIGHT + height, -height)
            alpha = bound(indv[i].get_alpha(), alphaMove, minAlpha, 1)
            newCircles.append(plt.Rectangle((centerX, centerY), width, height, color='black', alpha=alpha))

    if printFit: print(fitCount)    
    return newCircles

def new_generation(best, generationNum):
    population = [] # keep the original
    for i in range(popSize):
        mutated = mutate(best, generationNum, True if i == 0 else False)
        population.append(mutated)
    return population


population = []
popSize = 2
indvSize = 4000
generations = 1000

population = init_pop(popSize, indvSize)

rank = []

fig, ax = plt.subplots()
plt.ion()

for g in range(generations):
    rank = []
    for indv in population:
        phenome = graph_image(indv)
        fit = fitness(phenome)
        rank.append((fit, indv))
        #print(fit)
    
    rank = sorted(rank, key=itemgetter(0))
    if g % 5 == 0:
        imgB = graph_image(rank[-1][1])
        img = Image.fromarray(np.uint8(imgB))
        img.save(imgname + str(g) + '.png')
    population = new_generation(rank[-1][1], g)
    #print(rank[-1][1])
