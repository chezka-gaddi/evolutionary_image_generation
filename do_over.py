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

imgArr = np.asarray(Image.open('squares.jpg'), dtype=np.uint8)

# If not a 2D grayscale image, convert to grayscale
if len(imgArr.shape) != 2:
    imgArr = imgArr[:,:,0]

WIDTH, HEIGHT = imgArr.shape
USE_CIRCLES = True

def make_circle(x, y, r, c, a):
    return plt.Circle((x, -y), r, color=c, alpha=a)

def make_square(x, y, width, height, c, a):
    return plt.Rectangle((x, y), width, height, color=c, alpha=a)

def squareFitness(square):
    x = square.get_x()
    y = square.get_y()
    w = square.get_width()
    h = square.get_height()
    o = int(255 - square.get_alpha() * 255)
    diff = 0

    for i in range(x, x+w):
        for j in range(y, y+h):
            if j >= 0 and j < WIDTH and i >= 0 and i < WIDTH:
                diff += o - imgArr[i][j]
            else:
                diff += 255
    diff = np.abs(diff) / (w*h)
    return 1 - diff / 255

def circleFitness(cir):
    x = int(cir.center[0])
    y = int(-cir.center[1])
    r = int(cir.get_radius())
    o = int(255 - cir.get_alpha() * 255)
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
    return 1 - diff / 255

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
    h = int(HEIGHT / 6)

    for p in range(popSize):
        indv = []
        for i in range(indvSize):
            x, y = random.randint(5, WIDTH-5), random.randint(5, HEIGHT-5)
            if USE_CIRCLES:
                radius = random.randint(2, 3)
                a = np.random.rand()
                indv.append(make_circle(x, y, radius, 'k', a))
            else:
                width = random.randint(2, 6)
                height = random.randint(2, 6)
                a = np.random.rand()
                indv.append(make_square(x, y, width, height, 'k', a))

        population.append(indv)

    return population

def mutate(indvidual, generationNum):
    maxMovementChange = WIDTH / 2
    maxRadiusChange = 1
    maxOpacityChange = 0.4
    minRadius = 2
    maxRadius = 3

    newCircles = []
    for i in range(len(indv)):
        fitness = 0
        if USE_CIRCLES:
            fitness = circleFitness(indv[i])
        else:
            fitness = squareFitness(indv[i])

        curviness = 1.1
        crossover = 0.9
        maxMutationAmount = 1#(fitness*curviness - crossover*curviness)/(fitness - crossover*curviness)
        if fitness >= crossover: maxMutationAmount = 0
        print(fitness)
        print(maxMutationAmount)
        print("----")
        xMutateDirection = 1 if np.random.rand() > 0.5 else -1
        yMutateDirection = 1 if np.random.rand() > 0.5 else -1
        alphaMutateDirection = 1 if np.random.rand() > 0.5 else -1
        radiusMutateDirection = 1 if np.random.rand() > 0.5 else -1

        radiusMove =  np.random.rand() * maxRadiusChange * maxMutationAmount * radiusMutateDirection
        if indv[i].get_radius() + radiusMove > maxRadius or indv[i].get_radius() + radiusMove < minRadius:
            radiusMove *= -1
        radius = indv[i].get_radius() + radiusMove
        
        xMove = maxMovementChange * np.random.rand() * maxMutationAmount * xMutateDirection
        if indv[i].center[0] + xMove < radius or indv[i].center[0] + xMove > WIDTH - radius:
            xMove *= -1
        yMove = maxMovementChange * np.random.rand() * maxMutationAmount * yMutateDirection
        if indv[i].center[1] + yMove > -radius or indv[i].center[1] + yMove < -HEIGHT + radius:
            yMove *= -1

        center = (indv[i].center[0] + xMove, indv[i].center[1] + yMove)

        alpha = indv[i].get_alpha()

        newCircles.append(plt.Circle((center[0], center[1]), radius, color=indv[i].get_edgecolor(), alpha=alpha))
        
    return newCircles

def new_generation(best, generationNum):
    population = [] # keep the original
    for i in range(popSize - 1):
        mutated = mutate(best, generationNum)
        population.append(mutated)
    return population


population = []
popSize = 2
indvSize = 2000
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
        print(fit)
    print(g)
    
    rank = sorted(rank, key=itemgetter(0))
    if g % 5 == 0:
        imgB = graph_image(rank[-1][1])
        img = Image.fromarray(np.uint8(imgB))
        img.save('pic' + str(g) + '.png')
    population = new_generation(rank[-1][1], g)
    #print(rank[-1][1])
