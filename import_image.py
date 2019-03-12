from PIL import Image
import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
from operator import itemgetter

imgArr = np.asarray(Image.open('eye.png'))
#imgArr = imgArr[:,:,0]
WIDTH = len(imgArr)
HEIGHT = len(imgArr[0])

def make_circle(x, y, r, op):
    circle = plt.Circle((x, -y), r, color='black', alpha=op)
    return circle

def graph_image(circles):
    plt.clf()
    plt.axis((0, WIDTH, 0, -HEIGHT))
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.axis('off')
    ax = plt.gca()

    for c in circles:
        newPatch = copy.copy(c)
        ax.add_patch(newPatch)

    plt.get_current_fig_manager().resize(HEIGHT, WIDTH)
    plt.subplots_adjust(0,0,1,1,0,0)
    plt.ion()
    plt.gray()
    plt.show()
    plt.pause(.005)

    fig = plt.gcf()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    gray = data[:,:,0]

    if gray.shape == imgArr.shape:
        diff = abs(imgArr - gray)
        fit = 1/(WIDTH*HEIGHT) * np.sum(diff)

        genomes.append((fit, circles))

def init_pop(popSize, individualSize):
    population = []

    for j in range(popSize):
        circles = []
        h1 = int(HEIGHT / 20)
        h2 = int(HEIGHT * 0.75)

        for i in range(individualSize):
            x, y = random.randint(1, WIDTH), random.randint(1, HEIGHT)
            radius = random.randint(h1, h2)
            circles.append(make_circle(x, y, radius, np.random.rand()))

        population.append(circles)

    return population

def new_generation(motherGene, popSize):
    population = []

    for j in range(popSize):
        circles = []
        circles = circles + motherGene
        h1 = int(HEIGHT / 20)
        h2 = int(HEIGHT * .75)

        x, y = random.randint(1, WIDTH), random.randint(1, HEIGHT)
        radius = random.randint(h1, h2)
        circles.append(make_circle(x, y, radius, np.random.rand()))

        population.append(circles)

    return population

individualSize = 1
popSize = 100
generations = 1000

population = init_pop(popSize, individualSize)

radSize = HEIGHT
for g in range(generations):
    genomes = []

    for p in population:
        graph_image(p)

    genomes = sorted(genomes, key=itemgetter(0))

    print(genomes[0][0])
    population = new_generation(genomes[0][1], popSize)



def circleFitness(circleIndividual, img):
    count = 0
    total = 0
    for x in range(WIDTH):
        for y in range(HEIGHT):
            total += abs(circleCheck(circleIndividual, (x, y)) - img[x][y][0]/255)
            count += 1
    return total / count

def circleCheck(circleIndividual, point):
    val = 0
    for circle in circleIndividual:
        if np.sqrt((point[0] - circle[0]) ** 2 + (point[1] - circle[1]) ** 2) < circle[2]:
            val += (1 - val) * circle[3]
    return val

def triangleCheck(triangleIndividual, point):
    pass

def squareCheck(squareIndividual, point):
    pass
