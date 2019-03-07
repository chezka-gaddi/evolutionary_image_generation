from PIL import Image
import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import random

img_pop = []
imgArr = np.asarray(Image.open('panda.jpg'))
WIDTH = len(imgArr)
HEIGHT = len(imgArr[0])

def make_circle(x, y, r, op):
    circle = plt.Circle((x, -y), r, color='black', alpha=op)
    return circle

def graph_image(circles):
    plt.clf()
    plt.axis((0, WIDTH, 0, -HEIGHT))
    plt.gca().invert_yaxis()
    plt.axis('off')
    ax = plt.gca()

    colors = [100]*len(circles)
    c = PatchCollection(circles, alpha=0.1)
    c.set_array(np.array(colors))
    ax.add_collection(c)
    #for c in circles:
    #    ax.add_patch(c)

    plt.ion()
    plt.show()
    fig = plt.gcf()
    plt.pause(.005)
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

def init_pop(popSize, individualSize):
    population = []

    for j in range(popSize):
        circles = []

        for i in range(individualSize):
            x, y = random.randint(1,WIDTH), random.randint(1, HEIGHT)
            radius = random.randint(1, HEIGHT*0.15)
            circles.append(make_circle(x, y, radius, np.random.rand()))

        population.append(circles)

    return population

def fitness(arr, img):
    total = 0

    for x in range(480):
        for y in range(480):
            total = total + abs(arr[x][y] - img[x][y])

    print(total)


individualSize = 100
popSize = 5
generations = 1

population = init_pop(popSize, individualSize)
#population = sorted()

for g in range(generations):
    fit = []

    for q in population:
        graph_image(q)

#Image.fromarray(np.uint8(imgArr)).show()

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
