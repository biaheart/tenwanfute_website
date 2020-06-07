'''
数据预处理函数，输入文件内容异常会抛出error，存储文件的位置:'../data/original_data'
输入值：文件的相对或者绝对路径
返回值：成功读取返回'数据读取成功'
输出值：一个命名为输入文件创始者，存有MVA_BASE,BUS_NAMES,BUS_DATA,BRANCH_DATA的h5文件,注意BUS_NAMES的编码，是byte，不是unicode
'''


def read_data(file_path):
    import re
    import numpy as np
    import h5py
    storage_path = file_path[0:re.match('\.[^\.]+',file_path).end()] + '.h5'
    # 存储位置为源文件的同级目录
    with open(file_path, 'r') as f:
        flag = 0
        for line in f:  # next后迭代器会+1，每一次for循环迭代器也会加一
            if re.match(r'\s*[0-9]+/', line):                                                   # 匹配第一行
                mva_base_match = re.search(r'\s+[0-9]+.[0-9]*\s+', line)                        # 搜索唯一的浮点数
                mva_base = float(line[mva_base_match.start():mva_base_match.end()])             # 记录该浮点数
                data = h5py.File(storage_path, 'w')                                             # 创建以name为名称的h5文件存储数据
                data.create_dataset('MVA_BASE', data=mva_base)                                  # 存储基准功率
            if re.match('BUS', line):                                                           # 匹配母线部分
                bus_items_match = re.search(r'\s+[0-9]+\s+', line)                              # 搜索item
                bus_items = int(line[bus_items_match.start():bus_items_match.end()])            # 记录item
                bus = np.zeros((bus_items, 16))                                                 # 记录母线部分名字后面所有数据
                temp_names = list()
                for i in range(bus_items):                                                      # 读取bus数值
                    line = next(f)
                    bus[i, :] = list(map(float, line[18:].split()))
                    temp_names.append(line[5:17])
                bus_names = np.array(temp_names, dtype='S12')                                   # 转换list成为np数组
                data.create_dataset('BUS_NAMES', data=bus_names)                                # 存储母线名称
                data.create_dataset('BUS_DATA', data=bus)                                       # 存储母线数据
                next(f)
            if re.match('BRANCH', line):                                                        # 匹配支路部分
                branch_items_match = re.search(r'\s+[0-9]+\s+', line)                           # 搜索item
                branch_items = int(line[branch_items_match.start():branch_items_match.end()])   # 记录item
                branch = np.zeros((branch_items, 21))                                           # 记录支路部分所有数据
                for i in range(branch_items):
                    line = next(f)
                    branch[i, :] = list(map(float, line.split()))
                data.create_dataset('BRANCH_DATA', data=branch)                                 # 存储支路数据
                data.close()
                next(f)
                flag = 1                                                                        # 文件读取成功标志
                return storage_path
        assert flag, '文件不匹配'                                            # 判断是否导入正确的文件

'''
import h5py
file_name = r'IEEE30BusSystemCDF.txt'
read_data(file_name)
f = h5py.File(r'../data/original_data/UW ARCHIVE.h5', 'r')     # 打开一个h5文件对象
f['BUS_DATA'][(0, 1)]                   # 读取BUS_DATA的第一行第二列的数据
print(f.keys())
print(f['BRANCH_DATA'][()])
print(f['BUS_DATA'][()])
print(f['MVA_BASE'][()])
print(f['BUS_NAMES'][()])
f.close()
'''
