def load_flow_calculation(admatrix, a1, bus_num, MVA_BASE):     # 潮流计算函数
    '''
    :param admatrix: 节点导纳矩阵
    :param a1: 读取主键为''BUS_NAMES'的数据
    :param bus_num: 母线个数
    :param MVA_BASE: 基准功率
    :return: 电压，相角和节点注入功率
    '''
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
        dP1 = np.zeros((Bp.shape[0], 1))
        dQ1 = np.zeros((Bpp.shape[0], 1))
        dP1 = dP / U1
        dQ2 = dQ / U2
        U = -Bp.copy()  # 将雅可比矩阵值复制于上三角矩阵中
        a = Bp.shape[0]
        n = 0
        i = 1
        D = np.zeros((a, a), dtype=np.float)
        L = np.zeros((a, a), dtype=np.float)
        Z = np.zeros((a, a), dtype=np.float)

        while n < a:  # 消去n号节点
            k = n
            j = i - 1  # 通过中间变量i使得已知零元素不参与计算
            p = i - 1
            D[n][n] = U[n][n]  # 规格化矩阵元素生成
            if k == n:
                while p < a:
                    U[n][p] = U[n][p] / D[n][n]
                    Z[p][n] = U[p][n]
                    p = p + 1
            while j >= n and j < a:
                if U[n][j] != 0:  # 仅非零元素列需要参与计算
                    k = n + 1
                    while k < a:
                        U[k][j] = U[k][j] - Z[k][n] * U[n][j] / U[n][n]  # 形成上三角矩阵
                        k = k + 1
                j = j + 1
            n = n + 1
            i = i + 1
        L = U.transpose()
        first = np.zeros((a, 1), dtype=np.float)
        second = np.zeros((a, 1), dtype=np.float)
        dangle1 = np.zeros((a, 1), dtype=np.float)

        # 前代过程
        d = 1
        sum = 0
        first[0][0] = dP1[0][0]
        while d < a:
            e = 0
            while e < d:
                sum = sum + L[d][e] * first[e][0]
                e = e + 1
            first[d][0] = dP1[d][0] - sum
            sum = 0
            d = d + 1

        # 规格化过程
        t = 0
        while t < a:
            second[t][0] = first[t][0] / D[t][t]
            t = t + 1
        # 回代过程
        g = 2
        sum2 = 0
        dangle1[a - 1][0] = second[a - 1][0]
        while g < a + 1:
            h = 1
            while h < g:
                sum2 = sum2 + U[a - g][a - h] * dangle1[a - h][0]
                h = h + 1
            dangle1[a - g][0] = second[a - g][0] - sum2
            sum2 = 0
            g = g + 1
        dangle = dangle1 / U1  # 求解角度的不平衡量
        U = -Bpp.copy()  # 将雅可比矩阵值复制于上三角矩阵中
        a = Bpp.shape[0]
        n = 0
        i = 1
        D = np.zeros((a, a), dtype=np.float)
        L = np.zeros((a, a), dtype=np.float)
        Z = np.zeros((a, a), dtype=np.float)

        while n < a:  # 消去n号节点
            k = n
            j = i - 1  # 通过中间变量i使得已知零元素不参与计算
            p = i - 1
            D[n][n] = U[n][n]  # 规格化矩阵元素生成
            if k == n:
                while p < a:
                    U[n][p] = U[n][p] / D[n][n]
                    Z[p][n] = U[p][n]
                    p = p + 1
            while j >= n and j < a:
                if U[n][j] != 0:  # 仅非零元素列需要参与计算
                    k = n + 1
                    while k < a:
                        U[k][j] = U[k][j] - Z[k][n] * U[n][j] / U[n][n]  # 形成上三角矩阵
                        k = k + 1
                j = j + 1
            n = n + 1
            i = i + 1
        L = U.transpose()
        first = np.zeros((a, 1), dtype=np.float)
        second = np.zeros((a, 1), dtype=np.float)
        dU = np.zeros((a, 1), dtype=np.float)
        dQ2 = np.zeros((a, 1), dtype=np.float)

        # 前代过程
        d = 1
        sum = 0
        dQ2 = dQ / U2
        first[0][0] = dQ2[0][0]
        while d < a:
            e = 0
            while e < d:
                sum = sum + L[d][e] * first[e][0]
                e = e + 1
            first[d][0] = dQ2[d][0] - sum
            sum = 0
            d = d + 1

        # 规格化过程
        t = 0
        while t < a:
            second[t][0] = first[t][0] / D[t][t]
            t = t + 1
        # 回代过程
        g = 2
        sum2 = 0
        dU[a - 1][0] = second[a - 1][0]
        while g < a + 1:
            h = 1
            while h < g:
                sum2 = sum2 + U[a - g][a - h] * dU[a - h][0]
                h = h + 1
            dU[a - g][0] = second[a - g][0] - sum2
            sum2 = 0
            g = g + 1
        # 求解PQ节点电压不平衡量
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
        U = -J.copy()  # 将雅可比矩阵值复制于上三角矩阵中
        a = Bp.shape[0] + Bpp.shape[0]
        n = 0
        i = 1
        D = np.zeros((a, a), dtype=np.float)
        L = np.zeros((a, a), dtype=np.float)
        Z = np.zeros((a, a), dtype=np.float)

        while n < a:  # 消去n号节点
            k = n
            j = i - 1  # 通过中间变量i使得已知零元素不参与计算
            p = i - 1
            D[n][n] = U[n][n]  # 规格化矩阵元素生成
            if k == n:
                while p < a:
                    U[n][p] = U[n][p] / D[n][n]
                    Z[p][n] = U[p][n]
                    L[p][n] = Z[p][n] / D[n][n]
                    L[n][n] = U[n][n]
                    p = p + 1

            while j >= n and j < a:
                if U[n][j] != 0:  # 仅非零元素列需要参与计算
                    k = n + 1
                    while k < a:
                        U[k][j] = U[k][j] - Z[k][n] * U[n][j] / U[n][n]  # 形成上三角矩阵
                        k = k + 1
                j = j + 1
            n = n + 1
            i = i + 1

        first = np.zeros((a, 1), dtype=np.float)
        second = np.zeros((a, 1), dtype=np.float)
        d_unknowns = np.zeros((a, 1), dtype=np.float)

        # 前代过程
        d = 1
        sum = 0

        first[0][0] = unbalanced_matrix[0][0]
        while d < a:
            e = 0
            while e < d:
                sum = sum + L[d][e] * first[e][0]
                e = e + 1
            first[d][0] = unbalanced_matrix[d][0] - sum
            sum = 0
            d = d + 1

        # 规格化过程
        t = 0
        while t < a:
            second[t][0] = first[t][0] / D[t][t]
            t = t + 1
        # 回代过程
        g = 2
        sum2 = 0
        d_unknowns[a - 1][0] = second[a - 1][0]
        while g < a + 1:
            h = 1
            while h < g:
                sum2 = sum2 + U[a - g][a - h] * d_unknowns[a - h][0]
                h = h + 1
            d_unknowns[a - g][0] = second[a - g][0] - sum2
            sum2 = 0
            g = g + 1
        
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
    S_actual_value = np.zeros((1, bus_num), dtype=np.complex)   # 计算各节点注入功率有名值
    while i < bus_num:
        P_i = 0
        Q_i = 0
        while j < bus_num:  # 计算i节点注入有功的标幺值
            angle_ij_final = initialize[1, i] - initialize[1, j]
            P_i += initialize[0, i] * initialize[0, j] * (
                    RE[i, j] * math.cos(angle_ij_final) + IM[i, j] * math.sin(angle_ij_final))
            j = j + 1
        j = 0
        while j < bus_num:  # 计算i节点注入无功的标幺值
            angle_ij_final = initialize[1, i] - initialize[1, j]
            Q_i += initialize[0, i] * initialize[0, j] * (
                    RE[i, j] * math.sin(angle_ij_final) - IM[i, j] * math.cos(angle_ij_final))
            j = j + 1
        j = 0
        S_actual_value[0, i] = complex(P_i * MVA_BASE, Q_i * MVA_BASE)   # 计算平衡节点注入功率的有名值
        i = i + 1
    return U_actual_value, angle_actual_value, S_actual_value



