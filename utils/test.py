import pandas as pd
import numpy as np
import os


'''
# groupby()多标签归类
df1 = pd.DataFrame({"col1":list("ababbc"),
                   "col2":list("xxyyzz"),
                   "number1":range(90,96),
                   "number2":range(100,106)})
print(df1)
df2 = df1.groupby(["col1","col2"]).agg({"number1":sum,
                                        "number2":np.mean})

print(df2)
'''



allFileNum = 0
allFileList = []  # 存放 当前路径 以及当前路径的子路径 下的所有文件


def getAllFilesInPath(path):
    global allFileNum
    curPathDirList = []  # 当前路径下的所有文件夹
    files = os.listdir(path)  # 返回当前路径下的所有文件和文件夹
    for f in files:
        if os.path.isdir(path + "/" + f):
            if f[0] == ".":
                pass  # 排除隐藏文件夹
            else:
                curPathDirList.append(f)  # 添加非隐藏文件夹
        if os.path.isfile(path + "/" + f):
            allFileList.append(f)  # 添加文件
            allFileNum = allFileNum + 1  # 总文件数+1
            comparePackageName(f)
    for dl in curPathDirList:
        getAllFilesInPath(path + "/" + dl)  # 递归获取当前目录下的文件夹内的文件


def comparePackageName(f):
    absPath = os.path.abspath(f)
    print(absPath)


if __name__ == '__main__':
    getAllFilesInPath("/Users/wangyuan/AndroidStudioProjects")
    print("当前路径下的总文件数 =", allFileNum)
