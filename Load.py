import pandas
from os import path

students = ["ANDRIJA", "ACA", "NINA"]
basePath = path.dirname(__file__)
filePath = path.abspath(path.join(basePath, "..", "result", "ANDRIJA", "p1", "User 2_all_gaze.csv"))

#print(df.dtypes)
#axes = df.axes
#print(axes[1]) #axes[1][1] prvi red

def toRegion(filePath):
  df = pandas.read_csv(filePath)
  #print(df.dtypes)
  axes = df.axes
  print(axes[1]) #axes[1][1] prvi red

  


#print(str(df.loc[0][0]))
