

def load_flow_calculation(admatrix, a1, bus_num, MVA_BASE):     # 潮流计算函数
    import numpy as np
    import math
    RE = np.zeros((bus_num, bus_num))  # 实部矩阵
    IM = np.zeros((bus_num, bus_num))  # 虚部矩阵
    i = 0
    j = 0
    while i < bus_num:  # 获取实部虚部矩阵
        while j < bus_num:
            RE[i, j] = admatrix[i, j].real
            IM[i, j] = admatrix[i, j].imag
            j = j + 1
        i = i + 1
        j = 0

    i = bus_num - 1
    Bp = IM  # 生成B撇矩阵
    while i >= 0:
        if a1[i, 2] == 3:
            Bp = np.delete(Bp, i, axis=1)
            Bp = np.delete(Bp, i, axis=0)
            inU = a1[i, 3]  # 获取平衡节点电压作为初始标幺值
            inD = a1[i, 4] * math.pi / 180  # 获取平衡节点相角作为初值,并转换成弧度制
        i = i - 1
    j = bus_num - 1
    Bpp = IM  # 生成B撇撇矩阵
    while j >= 0:
        if a1[j, 2] != 0:
            Bpp = np.delete(Bpp, j, axis=1)
            Bpp = np.delete(Bpp, j, axis=0)
        j = j - 1

    initialize = np.zeros((4, bus_num), dtype=np.float)
    # 建立存储电压和相角,节点有功（平衡节点显示为0）和PQ节点无功（PV节点及平衡节点显示为0）的矩阵
    i = 0
    while i < bus_num:
        if a1[i, 2] == 0 or a1[i, 2] == 3:  # PQ节点和平衡节点的电压、相角，PQ节点的P、Q初始化
            initialize[0, i] = inU   # 电压标幺值
            initialize[1, i] = inD
            if a1[i, 2] == 0:
                initialize[2, i] = (a1[i, 7] - a1[i, 5]) / MVA_BASE  # 注入有功标幺值
                initialize[3, i] = (a1[i, 8] - a1[i, 6]) / MVA_BASE  # 注入无功标幺值
        if a1[i, 2] == 2:  # PV节点的初始化
            initialize[0, i] = a1[i, 3]   # 电压标幺值
            initialize[1, i] = inD        # 相角
            initialize[2, i] = (a1[i, 7] - a1[i, 5]) / MVA_BASE  # 注入有功标幺值
        i = i + 1
    # 注意：非平衡节点个数:Bp.shape[0] PQ节点个数:Bpp.shape[0]

    # PQ分解法迭代，收敛程度<0.01
    dP = np.zeros((Bp.shape[0], 1))  # 建立P,Q不平衡量矩阵
    dQ = np.zeros((Bpp.shape[0], 1))
    k = 0       # k为PQ分解法迭代次数
    while 1:   # 迭代
        if np.max(dP) < 0.01 and np.max(dQ) < 0.01 and k > 0:
            break
        i = 0
        j = 0
        m = 0
        while i < bus_num:  # 计算P的不衡量矩阵（列向量）
            if a1[i, 2] != 3:
                Pi = 0
                while j < bus_num:
                    angleij = initialize[1, i] - initialize[1, j]
                    Pi += initialize[0, i] * initialize[0, j] * (RE[i, j] * math.cos(angleij) + IM[i, j] * math.sin(angleij))
                    j = j + 1
                dP[m, 0] = initialize[2, i] - Pi
                m = m + 1
                j = 0
            i = i + 1
        i = 0
        j = 0
        m = 0
        while i < bus_num:  # 计算Q不平衡矩阵（列向量）
            if a1[i, 2] == 0:
                Qi = 0
                while j < bus_num:
                    angleij = initialize[1, i] - initialize[1, j]
                    Qi += initialize[0, i] * initialize[0, j] * (RE[i, j] * math.sin(angleij) - IM[i, j] * math.cos(angleij))
                    j = j + 1
                dQ[m, 0] = initialize[3, i] - Qi
                m = m + 1
                j = 0
            i = i + 1
        U1 = np.zeros((Bp.shape[0], 1))     # 定义除去平衡节点的电压列向量矩阵
        U2 = np.zeros((Bpp.shape[0], 1))    # 定义除去平衡节点和PV节点的电压列向量矩阵
        i = 0
        m = 0
        while i < bus_num:      # 获取U1矩阵
            if a1[i, 2] != 3:
                U1[m, 0] = initialize[0, i]
                m = m + 1
            i = i + 1
        i = 0
        m = 0
        while i < bus_num:  # 获取U2矩阵
            if a1[i, 2] == 0:
                U2[m, 0] = initialize[0, i]
                m = m + 1
            i = i + 1
        dangle = np.linalg.solve(-Bp, dP / U1) / U1  # 求解角度的不平衡量
        dU = np.linalg.solve(-Bpp, dQ / U2)  # 求解PQ节点电压不平衡量
        i = 0
        m = 0
        while i < bus_num:  # 求解下一次循环使用的电压初值
            if a1[i, 2] == 0:
                initialize[0, i] = initialize[0, i] + dU[m, 0]
                m = m + 1
            i = i + 1
        i = 0
        m = 0
        while i < bus_num:  # 求解下一次循环使用的相角初值
            if a1[i, 2] != 3:
                initialize[1, i] = initialize[1, i] + dangle[m, 0]
                m = m + 1
            i = i + 1
        k = k + 1

    # 常规牛顿拉夫逊法，收敛程度<0.0000001
    dP_new = np.zeros((Bp.shape[0], 1))  # 建立常规牛拉法P,Q不平衡量矩阵
    dQ_new = np.zeros((Bpp.shape[0], 1))
    k_new = 0     # k_new为常规牛顿拉夫逊法的迭代次数
    while 1:    # 迭代
        i = 0
        j = 0
        m = 0
        while i < bus_num:  # 计算dP_new的不衡量矩阵（列向量）
            if a1[i, 2] != 3:
                Pi_new = 0
                while j < bus_num:
                    angleij_new = initialize[1, i] - initialize[1, j]
                    Pi_new += initialize[0, i] * initialize[0, j] * (
                                RE[i, j] * math.cos(angleij_new) + IM[i, j] * math.sin(angleij_new))
                    j = j + 1
                dP_new[m, 0] = initialize[2, i] - Pi_new
                m = m + 1
                j = 0
            i = i + 1
        i = 0
        j = 0
        m = 0
        while i < bus_num:  # 计算dQ_new不平衡矩阵（列向量）
            if a1[i, 2] == 0:
                Qi_new = 0
                while j < bus_num:
                    angleij_new = initialize[1, i] - initialize[1, j]
                    Qi_new += initialize[0, i] * initialize[0, j] * (
                                RE[i, j] * math.sin(angleij_new) - IM[i, j] * math.cos(angleij_new))
                    j = j + 1
                dQ_new[m, 0] = initialize[3, i] - Qi_new
                m = m + 1
                j = 0
            i = i + 1

        unbalanced_matrix = np.zeros((Bp.shape[0]+Bpp.shape[0], 1))  # 生成完整的（n-m+1）×1阶的不平衡量矩阵
        i = 0
        j = 0
        while i < Bp.shape[0]:
            unbalanced_matrix[i, 0] = dP_new[j, 0]
            j = j + 1
            i = i + 1
        j = 0
        while i < Bp.shape[0]+Bpp.shape[0]:
            unbalanced_matrix[i, 0] = dQ_new[j, 0]
            j = j + 1
            i = i + 1

        if np.max(unbalanced_matrix) < 0.0000001:   # 判断收敛
            break

        H = np.zeros((Bp.shape[0], Bp.shape[0]))       # 初始化H，N，K，L矩阵
        N = np.zeros((Bp.shape[0], Bpp.shape[0]))
        K = np.zeros((Bpp.shape[0], Bp.shape[0]))
        L = np.zeros((Bpp.shape[0], Bpp.shape[0]))
        i = 0
        j = 0
        m = 0
        n = 0
        h = 0
        while i < bus_num:  # 计算H矩阵元素
            if a1[i, 2] != 3:
                while j < bus_num:
                    if a1[j, 2] != 3:
                        if m == n:   # 判断是H对角线上元素
                            Qi_H = 0
                            while h < bus_num:     # 计算H中对角线元素的注入无功功率一项
                                angle_ij = initialize[1, i] - initialize[1, h]
                                Qi_H += initialize[0, i] * initialize[0, h] * (
                                        RE[i, h] * math.sin(angle_ij) - IM[i, h] * math.cos(angle_ij))
                                h = h + 1
                            H[m, n] = initialize[0, i] * initialize[0, i] * IM[i, i] + Qi_H
                            h = 0
                        if m != n:   # 判断不是H对角线上元素
                            Angle_ij = initialize[1, i] - initialize[1, j]
                            H[m, n] = -initialize[0, i] * initialize[0, j] * (
                                        RE[i, j] * math.sin(Angle_ij) - IM[i, j] * math.cos(Angle_ij))
                        n = n + 1
                    j = j + 1
                m = m + 1
                j = 0
                n = 0
            i = i + 1

        i = 0
        j = 0
        m = 0
        n = 0
        h = 0
        while i < bus_num:  # 计算N矩阵元素
            if a1[i, 2] != 3:
                while j < bus_num:
                    if a1[j, 2] == 0:
                        if m == n:   # 判断是N对角线上元素
                            Pi_N = 0
                            while h < bus_num:     # 计算N中对角线元素的注入有功功率一项
                                angle_ij = initialize[1, i] - initialize[1, h]
                                Pi_N += initialize[0, i] * initialize[0, h] * (
                                        RE[i, h] * math.cos(angle_ij) + IM[i, h] * math.sin(angle_ij))
                                h = h + 1
                            N[m, n] = -initialize[0, i] * initialize[0, i] * RE[i, i] - Pi_N
                            h = 0
                        if m != n:   # 判断不是N对角线上元素
                            Angle_ij = initialize[1, i] - initialize[1, j]
                            N[m, n] = -initialize[0, i] * initialize[0, j] * (
                                        RE[i, j] * math.cos(Angle_ij) + IM[i, j] * math.sin(Angle_ij))
                        n = n + 1
                    j = j + 1
                m = m + 1
                j = 0
                n = 0
            i = i + 1

        i = 0
        j = 0
        m = 0
        n = 0
        h = 0
        while i < bus_num:  # 计算K矩阵元素
            if a1[i, 2] == 0:
                while j < bus_num:
                    if a1[j, 2] != 3:
                        if m == n:   # 判断是K对角线上元素
                            Pi_K = 0
                            while h < bus_num:  # 计算K中对角线元素的注入有功功率一项
                                angle_ij = initialize[1, i] - initialize[1, h]
                                Pi_K += initialize[0, i] * initialize[0, h] * (
                                        RE[i, h] * math.cos(angle_ij) + IM[i, h] * math.sin(angle_ij))
                                h = h + 1
                            K[m, n] = initialize[0, i] * initialize[0, i] * RE[i, i] - Pi_K
                            h = 0
                        if m != n:   # 判断不是K对角线上元素
                            Angle_ij = initialize[1, i] - initialize[1, j]
                            K[m, n] = initialize[0, i] * initialize[0, j] * (
                                        RE[i, j] * math.cos(Angle_ij) + IM[i, j] * math.sin(Angle_ij))
                        n = n + 1
                    j = j + 1
                m = m + 1
                j = 0
                n = 0
            i = i + 1

        i = 0
        j = 0
        m = 0
        n = 0
        h = 0
        while i < bus_num:  # 计算L矩阵元素
            if a1[i, 2] == 0:
                while j < bus_num:
                    if a1[j, 2] == 0:
                        if m == n:   # 判断是L对角线上元素
                            Qi_L = 0
                            while h < bus_num:  # 计算L中对角线元素的注入无功功率一项
                                angle_ij = initialize[1, i] - initialize[1, h]
                                Qi_L += initialize[0, i] * initialize[0, h] * (
                                        RE[i, h] * math.sin(angle_ij) - IM[i, h] * math.cos(angle_ij))
                                h = h + 1
                            L[m, n] = initialize[0, i] * initialize[0, i] * IM[i, i] - Qi_L
                            h = 0
                        if m != n:   # 判断不是L对角线上元素
                            Angle_ij = initialize[1, i] - initialize[1, j]
                            L[m, n] = -initialize[0, i] * initialize[0, j] * (
                                        RE[i, j] * math.sin(Angle_ij) - IM[i, j] * math.cos(Angle_ij))
                        n = n + 1
                    j = j + 1
                m = m + 1
                j = 0
                n = 0
            i = i + 1

        J = np.zeros((Bp.shape[0] + Bpp.shape[0], Bp.shape[0] + Bpp.shape[0]))     # 把H，N，K，L组合成J矩阵
        J[0:Bp.shape[0], 0:Bp.shape[0]] = H  # 导入H矩阵
        J[0:Bp.shape[0], Bp.shape[0]:Bp.shape[0] + Bpp.shape[0]] = N   # 导入N矩阵
        J[Bp.shape[0]:Bp.shape[0] + Bpp.shape[0], 0:Bp.shape[0]] = K   # 导入K矩阵
        J[Bp.shape[0]:Bp.shape[0] + Bpp.shape[0], Bp.shape[0]:Bp.shape[0] + Bpp.shape[0]] = L   # 导入L矩阵

        d_unknowns = np.linalg.solve(-J, unbalanced_matrix)   # 求解方程组
        d_angle = np.zeros((Bp.shape[0], 1))    # 定义相角修正量列向量
        d_angle[0:Bp.shape[0], 0] = d_unknowns[0:Bp.shape[0], 0]   # 得到相角修正量

        U2_new = np.zeros((Bpp.shape[0], 1))  # 定义除去平衡节点和PV节点的电压列向量矩阵
        i = 0
        m = 0
        while i < bus_num:  # 获取U2_new矩阵
            if a1[i, 2] == 0:
                U2_new[m, 0] = initialize[0, i]
                m = m + 1
            i = i + 1
        U2p_dU = np.zeros((Bpp.shape[0], 1))     # 定义U2_new列向量的逆乘上电压修正量的列向量
        U2p_dU[0:Bpp.shape[0], 0] = d_unknowns[Bp.shape[0]:Bp.shape[0] + Bpp.shape[0], 0]
        d_U = U2p_dU * U2_new   # 得到电压修正量

        i = 0
        m = 0
        while i < bus_num:  # 求解下一次循环使用的电压初值
            if a1[i, 2] == 0:
                initialize[0, i] = initialize[0, i] + d_U[m, 0]
                m = m + 1
            i = i + 1
        i = 0
        m = 0
        while i < bus_num:  # 求解下一次循环使用的相角初值
            if a1[i, 2] != 3:
                initialize[1, i] = initialize[1, i] + d_angle[m, 0]
                m = m + 1
            i = i + 1
        k_new = k_new + 1

    U_actual_value = np.zeros((1, bus_num))   # 最终各节点电压行向量的初始化
    i = 0   # 根据不同节点的基准电压计算所有节点的电压有名值
    while i < bus_num:
        U_actual_value[0, i] = initialize[0, i] * a1[i, 9]
        i = i + 1

    angle_actual_value = np.zeros((1, bus_num))   # 最终各节点相角行向量的初始化
    i = 0   # 根据相角弧度制计算相角角度值
    while i < bus_num:
        angle_actual_value[0, i] = initialize[1, i] * 180 / math.pi
        i = i + 1

    i = 0
    j = 0
    S_balanced = complex(0, 0)  # 初始化平衡节点注入功率
    while i < bus_num:
        if a1[i, 2] == 3:
            P_balanced = 0
            Q_balanced = 0
            while j < bus_num:   # 计算平衡节点注入有功的标幺值
                angle_ij_balanced = initialize[1, i] - initialize[1, j]
                P_balanced += initialize[0, i] * initialize[0, j] * (
                        RE[i, j] * math.cos(angle_ij_balanced) + IM[i, j] * math.sin(angle_ij_balanced))
                j = j + 1
            j = 0
            while j < bus_num:   # 计算平衡节点注入无功的标幺值
                angle_ij_balanced = initialize[1, i] - initialize[1, j]
                Q_balanced += initialize[0, i] * initialize[0, j] * (
                        RE[i, j] * math.sin(angle_ij_balanced) - IM[i, j] * math.cos(angle_ij_balanced))
                j = j + 1
            j = 0
            S_balanced = complex(P_balanced, Q_balanced) * MVA_BASE   # 计算平衡节点注入功率的有名值
        i = i + 1
    return U_actual_value, angle_actual_value, S_balanced


'''
import h5py
from lib_packages.Admittance_Matrix_Class import Admittancematrix


f = h5py.File('UW ARCHIVE.h5', 'r')    # 正确读取对应的h5文件
a1 = f['BUS_DATA'][()]  # 读取主键为‘BUS_NAMES’的数据
bus_num = f['BUS_NAMES'].shape[0]
a2 = f['BRANCH_DATA'][()]
MVA_BASE = f['MVA_BASE'][()]
matrix = Admittancematrix(bus_num)
matrix.generate_matrix(a2[:, 0], a2[:, 1], a2[:, 6], a2[:, 7], a1[:, 13], a1[:, 14])
# print(matrix.get_matrix())
admatrix = matrix.get_matrix()

print(load_flow_calculation(admatrix, a1, bus_num, MVA_BASE))
'''