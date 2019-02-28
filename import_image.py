from PIL import Image
import numpy as np

# loads in image into numpy array
imgArr = np.asarray(Image.open('panda.jpg'))

print(imgArr)
WIDTH = len(imgArr)
HEIGHT = len(imgArr[0])

#img = Image.fromarray(np.uint8(imgArr))
Image.fromarray(np.uint8(imgArr)).show()

individualSize = 50
popSize = 100
pop = []

for j in range(popSize):
	circles = []
	for i in range(individualSize):
		x,y = np.random.rand() * WIDTH, np.random.rand() * HEIGHT
		radius = min(WIDTH - x, x, HEIGHT - y, y)
		radius *= np.random.rand();
		circles.append((x, y, radius, np.random.rand()))
	pop.append(circles)


def circleFitness(circleIndividual, img):
	count = 0
	total = 0
	for x in range(WIDTH):
		for y in range(HEIGHT):
			total += abs(circleCheck(circleIndividual, (x, y)) - img[x][y][0]/255)
			count += 1
	return total / count


def circleCheck(circleIndividual, point):
	#circle = [x, y, radius, opacity]
	val = 0
	for circle in circleIndividual:
		if np.sqrt((point[0] - circle[0]) ** 2 + (point[1] - circle[1]) ** 2) < circle[2]:
			val += (1 - val) * circle[3]
	return val

def triangleCheck(triangleIndividual, point):
	pass

def squareCheck(squareIndividual, point):
	pass

print(circleFitness(pop[0], imgArr))