from PIL import Image
import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import random

img_pop = []
imgArr = np.asarray(Image.open('eye.png'))
WIDTH = len(imgArr)
HEIGHT = len(imgArr[0])

def fitness(arr, img):
    total = 0

    for x in range(WIDTH):
        for y in range(HEIGHT):
            total = total + abs(arr[x][y] - img[x][y])

    print(total)

def make_circle(x, y, r, op):
    circle = plt.Circle((x, -y), r, color='black', alpha=op)
    return circle

def graph_image(circles):
    plt.clf()
    plt.axis((0, WIDTH, 0, -HEIGHT))
    plt.gca().invert_yaxis()
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_aspect('equal')
    plt.axis('off')
    ax = plt.gca()

    colors = [0]*len(circles)
    c = PatchCollection(circles, alpha=0.1)
    c.set_array(np.array(colors))
    ax.add_collection(c)
    #for c in circles:
    #    ax.add_patch(c)

    plt.get_current_fig_manager().resize(WIDTH, HEIGHT)
    plt.subplots_adjust(0,0,1,1,0,0)
    plt.ion()
    plt.show()
    plt.pause(.01)

    fig = plt.gcf()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    print(fig.canvas.get_width_height())
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    print(len(data), len(data[0]))
    img_pop.append(Image.fromarray(np.uint8(data)))

def init_pop(popSize, individualSize):
    population = []

    for j in range(popSize):
        circles = []
        h = int(HEIGHT*0.15)

        for i in range(individualSize):
            x, y = random.randint(1, WIDTH), random.randint(1, HEIGHT)
            radius = random.randint(1, h)
            circles.append(make_circle(x, y, radius, np.random.rand()))

        population.append(circles)

    return population


individualSize = 100
popSize = 25
generations = 1

population = init_pop(popSize, individualSize)
#population = sorted()

for g in range(generations):
    fit = []

    for q in population:
        graph_image(q)

img = np.asarray(img_pop[0])
fitness(imgArr, img)
print(len(img), len(img[0]))

#img = Image.fromarray(np.uint8(img_pop[0]))
img_pop[0].save('pic.png')

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
