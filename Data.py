class Region:
  def __init__(self, x1, y1, x2, y2):
    if x1 > x2 or y2 > y1:
      print("Pogresno kreiran region")
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


class Data:
  def __init__(self, x, y, time, region=Region(0, 0, 0, 0)):
    self.x = x
    self.y = y
    self.time = time
    self.region = region

  def print(self):
    print("X:" + str(self.x) + " Y:" + str(self.y) + " Time:" + str(self.time) + " Region: " + str(self.region.print()))


R1 = Region(0, 0, 150, 200)
p1 = Data(15, 21, 0.5, R1)
p1.print()
