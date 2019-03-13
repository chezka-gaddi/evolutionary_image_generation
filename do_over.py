from PIL import Image
import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import random
from random import shuffle
import copy
from operator import itemgetter

imgArr = np.asarray(Image.open('squares.jpg'), dtype=np.uint8)

# If not a 2D grayscale image, convert to grayscale
if len(imgArr.shape) != 2:
    imgArr = imgArr[:,:,0]

WIDTH, HEIGHT = imgArr.shape

def make_circle(x, y, r, c, a):
    return plt.Circle((x, -y), r, color=c, alpha=a)

def fitness(genome):
    if genome.shape == imgArr.shape:
        diff = np.absolute(genome - imgArr)
        return 1 - (np.sum(diff) / (WIDTH * HEIGHT * 250))

    return -1

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
    plt.pause(.001)

    fig = plt.gcf()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data[:,:,0]

    return data

def minimize_f(x, y, radius):
    x1 = x - radius
    x2 = x + radius
    y1 = y - radius
    y2 = y + radius
    y3 = y1

    if x1 < 0: x1 = 0
    if x2 > WIDTH: x2 = WIDTH
    if y1 < 0: y1 = 0
    if y2 > HEIGHT: y2 = HEIGHT

def init_pop(popSize, indvSize):
    population = []
    h = int(HEIGHT / 6)

    for p in range(popSize):
        indv = []
        for i in range(indvSize):
            x, y = random.randint(1, WIDTH), random.randint(1, HEIGHT)
            radius = random.randint(1, h)
            c = random.choice(['k', 'w', 'k'])
            a = np.random.rand()/2
            indv.append(make_circle(x, y, radius, c, a))

        population.append(indv)

    return population

def mutate(indv):
    shuffle(indv)
    fourth = int(len(indv)/4)
    h = int(HEIGHT / 6)
    for i in range(fourth):
        x = random.randint(0, len(indv)-1)
        indv[x].alpha = np.random.rand()
        indv[x].set_radius(random.randint(1,h))
    return indv

def recombine(mom, dad):
    mid = int(len(mom)/2)
    child = mom[:mid] + dad[mid:]
    child = mutate(child)
    return child

def new_generation(ranked):
    population = []
    mid = int(len(ranked)/2)
    i = -1
    while abs(i) < mid:
        population.append(recombine(ranked[i][1], ranked[i-1][1]))
        population.append(ranked[i][1])
        population.append(ranked[i-1][1])
        population.append(recombine(ranked[i-1][1], ranked[i][1]))
        i = i - 2

    return population


population = []
popSize = 100
indvSize = 100
generations = 1000

population = init_pop(popSize, indvSize)

for g in range(generations):
    rank = []

    for indv in population:
        phenome = graph_image(indv)
        fit = fitness(phenome)
        rank.append((fit, indv))

    rank = sorted(rank, key=itemgetter(0))
    print(rank[-1][0])
    imgB = graph_image(rank[-1][1])
    img = Image.fromarray(np.uint8(imgB))
    img.save('pic.png')
    population = new_generation(rank)
