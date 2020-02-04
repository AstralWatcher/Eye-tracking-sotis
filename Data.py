class Region:
    def __init__(self, x1, y1, x2, y2):
        if x1 > x2 or y2 > y1:
            print("Pogresno kreiran region")
            print("X1:" + str(x1) + "X2:" + str(x2) + "Y1:" + str(y1) + "y2:" + str(y2))
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def print(self):
        return "X1:" + str(self.x1) + " Y1:" + str(self.y1) + " X2:" + str(self.x2) + " Y2:" + str(self.y2)

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def getX1(self):
        return self.x1

    def getX2(self):
        return self.x2

    def getY1(self):
        return self.y1

    def getY2(self):
        return self.y2


class Data:
    def __init__(self, x, y, time, region=Region(0, 0, 0, 0)):
        self.x = x
        self.y = y
        self.time = time
        self.region = region

    def print(self):
        print("X:" + str(self.x) + " Y:" + str(self.y) + " Time:" + str(self.time) + " Region: " + str(
            self.region.print()))

    def setRegion(self, region):
        self.region = region

    def getX(self):
        return self.x

    def getY(self):
        return self.Y

    def getTime(self):
        return self.time

    def getRegion(self):
        return self.region


class DataArray:
    def __init__(self):
        self.array = []

    # end = true adds to end
    def insert(self, data, end=True):
        at = 0
        if end:
            at = self.array.__len__()
        self.array.insert(at, data)

    def print(self):
        for i in range(0, self.array.__len__()):
            self.array[i].print()

    def get(self, elementAt):
        if elementAt >= self.array.__len__():
            return -1
        return self.array[elementAt]


if __name__ == "__main__":
    R1 = Region(0, 0, 150, 200)
    p1 = Data(15, 21, 0.5, R1)
    p2 = Data(22, 33, 0.7, R1)

    D = DataArray()
    D.insert(p1)
    D.insert(p2)
    D.print()

    # print(D.get(1).x)

    # p1.print()
