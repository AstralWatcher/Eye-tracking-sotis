import pandas
from os import path
from dateutil import parser
from Data import Region
from Data import Data
from Data import DataArray

arr = DataArray()

R1 = Region(0, 0, 150, 200)
R2 = Region(151, 201, 201, 251)
regions = [R1, R2]


def readData(filePath):
  df = pandas.read_csv(filePath)
  print(df.dtypes.__len__())
  axes = df.axes
  print(axes[1])  # axes[1][1] prvi red
  sstarttime = axes[1][0]
  print("Start " + str(sstarttime))

  ddate = parser.parse(sstarttime[5:len(sstarttime) - 1])
  print(ddate)

  for index, row in df.head().iterrows():  # bez head prolazi kroz sve
    print(index, row[0], row[1], row[2], row[3], row[4], row[5])
    newData = Data(row[0], row[2], row[3], None)
    arr.insert(newData)


students = ["ANDRIJA", "ACA", "NINA"]
basePath = path.dirname(__file__)
filePath = path.abspath(path.join(basePath, "..", "result", "ANDRIJA", "p1", "User 2_all_gaze.csv"))

readData(filePath, regions)

# print(str(df.loc[0][0]))
