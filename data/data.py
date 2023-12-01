import pandas as pd

path = "C:\\Users\\yulin\\Desktop\\SC实践\\数据文件\\sample.csv"
data = pd.read_csv(path)

# print(data['content'])
# print(data)
data.to_excel('C:\\Users\\yulin\\Desktop\\SC实践\\数据文件\\sample.xlsx')