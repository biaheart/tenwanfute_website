import numpy as np


class Admittancematrix:  # 建立节点导纳矩阵类

    def __init__(self, bus_num):  # 初始化
        self.bus_num = bus_num
        self.admittance_matrix = np.zeros((bus_num, bus_num), dtype=complex)  # 节点导纳矩阵
        self.part_matrix = np.zeros((bus_num, bus_num), dtype=complex)  # 辅助矩阵

    def set_self_ad(self, node, conductance, susceptance):
        admittance = complex(conductance, susceptance)
        node = int(node)
        self.part_matrix[node - 1][node - 1] = admittance
        self.admittance_matrix = self.admittance_matrix + self.part_matrix
        self.part_matrix = np.zeros((self.bus_num, self.bus_num), dtype=complex)

    def add(self, node1, node2, resistance, reactance):  # 增加支路的支路追加法
        if resistance == 0 and reactance == 0:
            return
        impedance = complex(resistance, reactance)  # 计算阻抗
        admittance = 1 / impedance  # 计算导纳
        node1 = int(node1)
        node2 = int(node2)
        if node1 > self.bus_num or node2 > self.bus_num:
            if node1 > node2:
                max_node = node1
            else:
                max_node = node2
            c1 = np.zeros(self.bus_num, dtype=complex)
            c2 = np.zeros(max_node, dtype=complex)
            k = self.bus_num
            while k < max_node:
                self.admittance_matrix = np.row_stack((self.admittance_matrix, c1))
                self.part_matrix = np.row_stack((self.part_matrix, c1))
                k = k + 1
            k = self.bus_num
            while k < max_node:
                self.admittance_matrix = np.column_stack((self.admittance_matrix, c2))
                self.part_matrix = np.column_stack((self.part_matrix, c2))
                k = k + 1
            self.bus_num = max_node
        if node1 != node2:  # 判断不是并联导纳
            self.part_matrix[node1 - 1][node1 - 1] = admittance  # 根据增加支路修改辅助矩阵元素
            self.part_matrix[node2 - 1][node2 - 1] = admittance
            self.part_matrix[node1 - 1][node2 - 1] = -admittance
            self.part_matrix[node2 - 1][node1 - 1] = -admittance
            self.admittance_matrix = self.admittance_matrix + self.part_matrix  # 支路追加
            self.part_matrix = np.zeros((self.bus_num, self.bus_num), dtype=complex)  # 辅助矩阵置零
        else:  # 并联导纳
            self.part_matrix[node1 - 1][node1 - 1] = admittance
            self.admittance_matrix = self.admittance_matrix + self.part_matrix
            self.part_matrix[node1 - 1][node1 - 1] = 0

    def minus(self, node1, node2, resistance, reactance):  # 删除支路的支路追加法
        if resistance == 0 and reactance == 0:
            return
        impedance = complex(resistance, reactance)  # 计算阻抗
        admittance = 1 / impedance  # 计算导纳
        node1 = int(node1)
        node2 = int(node2)
        if node1 != node2:  # 判断不是并联导纳
            self.part_matrix[node1 - 1][node1 - 1] = -admittance  # 根据增加支路修改辅助矩阵元素
            self.part_matrix[node2 - 1][node2 - 1] = -admittance
            self.part_matrix[node1 - 1][node2 - 1] = admittance
            self.part_matrix[node2 - 1][node1 - 1] = admittance
            self.admittance_matrix = self.admittance_matrix + self.part_matrix  # 支路追加
            self.part_matrix = np.zeros((self.bus_num, self.bus_num), dtype=complex)  # 辅助矩阵置零
        else:  # 并联导纳
            self.part_matrix[node1 - 1][node1 - 1] = -admittance
            self.admittance_matrix = self.admittance_matrix + self.part_matrix
            self.part_matrix[node1 - 1][node1 - 1] = 0

    def get_matrix(self):
        return self.admittance_matrix

    def generate_matrix(self, nodelist1, nodelist2, resistance, reactance, conductance, susceptance):
        for i in range(len(nodelist1)):
            self.add(nodelist1[i], nodelist2[i], resistance[i], reactance[i])
        for i in range(self.bus_num):
            self.set_self_ad(i+1, conductance[i], susceptance[i])

'''
import h5py

f = h5py.File('UW ARCHIVE.h5', 'r')
a1 = f['BUS_DATA'][()]  # 读取主键为‘BUS_NAMES’的数据
bus_num = f['BUS_NAMES'].shape[0]
a2 = f['BRANCH_DATA'][()]
matrix = Admittancematrix(bus_num)
matrix.generate_matrix(a2[:, 0], a2[:, 1], a2[:, 6], a2[:, 7], a1[:, 13], a1[:, 14])
print(matrix.get_matrix())
'''