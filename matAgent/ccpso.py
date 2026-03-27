def run_once(self, action):
    # 1. 【核心修正】：遵循导师指导，剥离 RL 对 w, c1, c2 的控制权
    # 假设 action 是一维数组，只取第一个维度作为收敛控制总阀门 Conv_a
    # 映射到 [0.1, 1.5] 区间
    if isinstance(action, np.ndarray) or isinstance(action, list):
        Conv_a = action[0] * 0.7 + 0.8
    else:
        Conv_a = action * 0.7 + 0.8  # 兼容 action 传入纯标量的情况

    # 2. 【建立稳固的数学底座】：采用纯数学领域的黄金收敛常数
    c1_fixed = 1.49445
    c2_fixed = 1.49445
    # 惯性权重 w 随迭代进度线性递减 [0.9 -> 0.4]
    w_fixed = 0.9 - 0.5 * (self.fe_num / self.fe_max)

    # 生成全种群、全维度的随机扰动矩阵 (n_part, n_dim)
    r1 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
    r2 = np.random.uniform(0, 1, (self.n_part, self.n_dim))

    # 计算真实的、带随机数的引力项矩阵
    c1_r1 = c1_fixed * r1
    c2_r2 = c2_fixed * r2

    # 总引力矩阵 C_gravity
    C_gravity = c1_r1 + c2_r2

    # 3. 计算严谨的等效引力中心矩阵 Q (为避免除零加上 1e-16)
    # 注意：self.g_best 会利用 numpy 的广播机制自动扩充维度匹配
    Q = (c1_r1 * self.p_best + c2_r2 * self.g_best) / (C_gravity + 1e-16)

    # 4. 构建二阶差分方程的系数矩阵
    a1 = 1 + w_fixed - C_gravity
    a2 = -w_fixed

    # 5. 计算基于稳定底座的自然偏移量矩阵 X_Q
    X_Q = a1 * (self.xs - Q) + a2 * (self.xs_old - Q)

    # 6. 【核心降维打击】：唯一由强化学习干预的环节，通过相乘缩放自然位移
    new_xs = Q + Conv_a * X_Q

    # 7. 越界处理 (强行拉回搜索空间)
    new_xs = np.clip(new_xs, self.pos_min, self.pos_max)

    # 8. 状态迭代更替：时间步向前推进
    self.xs_old = self.xs.copy()  # 当前位置变成老位置 X(t-1)
    self.xs = new_xs  # 新位置更新为当前位置 X(t)

    # 9. 重新评估适应度并更新历史最佳
    self.fits = self.fun(self.xs)
    self.fe_num += self.n_part
    self.update_best()