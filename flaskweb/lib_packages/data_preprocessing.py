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
                    temp_names.append(line[5:17])
                    bus[i, 0] = float(line[18:20])
                    bus[i, 1] = float(line[20:23])
                    bus[i, 2] = float(line[24:26])
                    bus[i, 3] = float(line[27:33])
                    bus[i, 4] = float(line[33:40])
                    bus[i, 5] = float(line[40:49])
                    bus[i, 6] = float(line[49:59])
                    bus[i, 7] = float(line[59:67])
                    bus[i, 8] = float(line[67:75])
                    bus[i, 9] = float(line[76:83])
                    bus[i, 10] = float(line[84:90])
                    bus[i, 11] = float(line[90:98])
                    bus[i, 12] = float(line[98:106])
                    bus[i, 13] = float(line[106:114])
                    bus[i, 14] = float(line[114:122])
                    bus[i, 15] = float(line[123:127])
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
                    branch[i, 0] = float(line[0:4])
                    branch[i, 1] = float(line[5:9])
                    branch[i, 2] = float(line[10:12])
                    branch[i, 3] = float(line[13:15])
                    branch[i, 4] = float(line[16])
                    branch[i, 5] = float(line[18])
                    branch[i, 6] = float(line[19:29])
                    branch[i, 7] = float(line[29:40])
                    branch[i, 8] = float(line[40:50])
                    branch[i, 9] = float(line[50:55])
                    branch[i, 10] = float(line[56:62])
                    branch[i, 11] = float(line[62:69])
                    branch[i, 12] = float(line[68:72])
                    branch[i, 13] = float(line[73])
                    branch[i, 14] = float(line[76:82])
                    branch[i, 15] = float(line[83:90])
                    branch[i, 16] = float(line[90:97])
                    branch[i, 17] = float(line[97:104])
                    branch[i, 18] = float(line[105:111])
                    branch[i, 19] = float(line[112:118])
                    branch[i, 20] = float(line[118:126])
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
