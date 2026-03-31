# import numpy as np
# from matAgent.testpso import TestpsoSwarm  # 直接继承原版，复用它的 CLPSO/FDR 目标计算逻辑


# class FiftyDimCCPsoSwarm(TestpsoSwarm):
#     optimizer_name = 'CCPSO_50D'
#     action_space = 10
#     obs_space = 1

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.name = '50D_CC_PSO'
#         # 统一式特需：记录上一代的位置 X(t-1)
#         self.xs_old = self.xs.copy()

#     def run_once(self, actions):
#         if actions is None:
#             actions = np.zeros(self.action_space * self.n_group)
#         if self.show:
#             print('{}|best fit:{}'.format(self.fe_num / self.fe_max, self.history_best_fit))

#         new_xs = np.zeros_like(self.xs)

#         for i in range(self.n_part):
#             fdr_deta_fitness = self.atom_history_best_fits[i] - self.atom_history_best_fits

#             # 和训练时的 group 配置保持一致
#             action = actions[
#                 i % self.n_group * self.action_space:
#                 i % self.n_group * self.action_space + self.action_space
#             ]

#             w = action[7] * 0.4 + 0.5
#             r1 = action[1] * 1.5 + 1.5
#             r2 = action[2] * 1.5 + 1.5
#             r5 = action[5] * 1.5 + 1.5
#             r6 = action[6] * 1.5 + 1.5

#             # 【核心改动 1】：提取 action[0] 作为收敛性控制目标 P_ECon
#             P_ECon = action[0] * 0.5 + 0.5  # 映射到 [0, 1] 之间

#             r = np.array([r1, r2, 0, 0, r5, r6])
#             r = r / (np.sum(r) + 1e-10) * (action[8] + 1) * 4
#             r1, r2, r3, r4, r5, r6 = r

#             mutation_rate = (action[9] + 1) * 0.01

#             for d in range(self.n_dim):
#                 # 完全保留原版的异构目标寻找逻辑
#                 clpso_target = self.p_best[self.fid[i, d], d]

#                 xid = self.xs[i, d]
#                 distance = xid - self.p_best[:, d]
#                 distance[i] = np.inf
#                 fdr = fdr_deta_fitness / (distance + 1e-250)
#                 j_index = np.argmax(fdr)
#                 fdr_target = self.p_best[j_index, d]

#                 gbest_target = self.g_best[d]
#                 pbest_target = self.p_best[i, d]

#                 # 【核心改动 2】：计算总引力 C 和等效中心 Q
#                 C = r1 + r2 + r5 + r6

#                 # 计算目标点的加权中心
#                 Q = (r1 * clpso_target + r2 * fdr_target + r5 * gbest_target + r6 * pbest_target) / (C + 1e-16)

#                 # 计算统一式系数
#                 a1 = 1 + w - C
#                 a2 = -w

#                 # 计算自然期望偏差 X_Q
#                 X_Q = a1 * (self.xs[i, d] - Q) + a2 * (self.xs_old[i, d] - Q)

#                 # 计算一维的实际收敛度 P_Con (避免除零)
#                 P_Con = np.abs(X_Q)
#                 if P_Con == 0:
#                     P_Con = 1e-16

#                 # 应用收敛性控制公式更新当前维度的位置
#                 new_xs[i, d] = Q + (P_ECon / P_Con) * X_Q

#             # 保留原版的突变逻辑以求公平
#             if np.random.random() < mutation_rate * self.flag[i]:
#                 new_xs[i] = np.random.uniform(self.pos_min, self.pos_max, self.xs[i].shape)

#         # 越界处理
#         new_xs = np.clip(new_xs, self.pos_min, self.pos_max)

#         # 状态迭代
#         self.xs_old = self.xs.copy()
#         self.xs = new_xs

#         self.fits = self.fun(self.xs)
#         self.update_best()


####以下是输出四个参数的版本（2026.3.24与导师交流后暂不考虑，即使取得较好效果）######
# import numpy as np
# from matAgent.testpso import TestpsoSwarm  

# class FiftyDimCCPsoSwarm(TestpsoSwarm):
#     optimizer_name = 'CCPSO_50D'
#     action_space = 10
#     obs_space = 1

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.name = '50D_CC_PSO'
#         # 记录上一代的位置 X(t-1)
#         self.xs_old = self.xs.copy()

#     def run_once(self, actions):
#         # 兜底动作，防止 actions 为空
#         if actions is None:
#             actions = np.zeros(self.action_space * self.n_group)
            
#         if self.show:
#             print('{}|best fit:{}'.format(self.fe_num / self.fe_max, self.history_best_fit))

#         new_xs = np.zeros_like(self.xs)

#         for i in range(self.n_part):
#             # 提取动作 (网络仍然输出 10 个，但我们只取其中 4 个)
#             action = actions[
#                 i % self.n_group * self.action_space:
#                 i % self.n_group * self.action_space + self.action_space
#             ]

#             # 【决策 2：只提取 4 个核心参数，其余废弃】
#             # 1. 收敛性控制系数 Conv_a (映射到 [0.1, 1.5])
#             Conv_a = action[0] * 0.7 + 0.8  
#             # 2. 个体认知因子 c1 (映射到 [0.0, 3.0])
#             c1 = action[1] * 1.5 + 1.5      
#             # 3. 社会认知因子 c2 (映射到 [0.0, 3.0])
#             c2 = action[2] * 1.5 + 1.5      
#             # 4. 惯性权重 w (映射到 [0.1, 0.9])
#             w  = action[7] * 0.4 + 0.5      

#             for d in range(self.n_dim):
#                 # 【决策 1：严谨引入随机数 r1, r2】
#                 # 为每个粒子的每个维度生成独立的均匀分布随机数
#                 r1 = np.random.rand()
#                 r2 = np.random.rand()

#                 # 回归纯粹，只使用标准 PSO 的 pbest 和 gbest
#                 pbest_target = self.p_best[i, d]
#                 gbest_target = self.g_best[d]

#                 # 计算真实的、带随机数的引力项
#                 c1_r1 = c1 * r1
#                 c2_r2 = c2 * r2
                
#                 # 总引力 C_gravity
#                 C_gravity = c1_r1 + c2_r2  
                
#                 # 计算严谨的等效引力中心 Q
#                 Q = (c1_r1 * pbest_target + c2_r2 * gbest_target) / (C_gravity + 1e-16)

#                 # 构建二阶差分方程的系数
#                 a1 = 1 + w - C_gravity
#                 a2 = -w

#                 # 计算无干预下的自然偏移量 X_Q
#                 # X_Q = a1*( X(t-1) - Q ) + a2*( X(t-2) - Q )
#                 X_Q = a1 * (self.xs[i, d] - Q) + a2 * (self.xs_old[i, d] - Q)

#                 # 施加收敛性策略干预：X(t) = Q + Conv_a * X_Q
#                 new_xs[i, d] = Q + Conv_a * X_Q

#             # 【决策 3：已彻底删除所有 Mutation 变异代码，杜绝“开小灶”】

#         # 越界处理 (强行拉回搜索空间)
#         new_xs = np.clip(new_xs, self.pos_min, self.pos_max)

#         # 状态迭代：更新历史轨迹
#         self.xs_old = self.xs.copy() # 当前代变老一代 X(t-1) -> X(t-2)
#         self.xs = new_xs             # 新一代变当前代 X(t) -> X(t-1)

#         # 计算适应度并更新最优记录
#         self.fits = self.fun(self.xs)
#         self.update_best()


##########线性递增策略+收敛性策略###################
# import numpy as np
# from matAgent.testpso import TestpsoSwarm  

# class FiftyDimCCPsoSwarm(TestpsoSwarm):
#     optimizer_name = 'CCPSO_50D'
#     action_space = 10
#     obs_space = 1

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.name = '50D_CC_PSO'
#         # 记录上一代的位置 X(t-1)
#         self.xs_old = self.xs.copy()

#     def run_once(self, actions):
#         if actions is None:
#             actions = np.zeros(self.action_space * self.n_group)
            
#         if self.show:
#             print('{}|best fit:{}'.format(self.fe_num / self.fe_max, self.history_best_fit))

#         new_xs = np.zeros_like(self.xs)

#         # 【核心修正 1：保留底层 PSO 的动态生命力】
#         # 惯性权重 w 随进程动态线性递减 (0.9 -> 0.4)，复刻经典 PSO
#         w_dynamic = 0.9 - 0.5 * (self.fe_num / self.fe_max)
        
#         # 基础引力常数保留 1.49445
#         c1_base = 1.49445
#         c2_base = 1.49445

#         for i in range(self.n_part):
#             action = actions[
#                 i % self.n_group * self.action_space:
#                 i % self.n_group * self.action_space + self.action_space
#             ]

#             # 【核心修正 2：遵循导师指导，RL 只输出并控制总阀门 Conv_a】
#             # 映射到 [0.1, 1.5] 的区间
#             Conv_a = action[0] * 0.7 + 0.8  

#             for d in range(self.n_dim):
#                 # 每维独立生成随机扰动，保留种群多样性
#                 r1 = np.random.rand()
#                 r2 = np.random.rand()

#                 pbest_target = self.p_best[i, d]
#                 gbest_target = self.g_best[d]

#                 # 动态随机引力
#                 c1_r1 = c1_base * r1
#                 c2_r2 = c2_base * r2
                
#                 # 总引力 C_gravity (每一维、每一步都在自然变化)
#                 C_gravity = c1_r1 + c2_r2  
                
#                 # 动态加权目标中心 Q
#                 Q = (c1_r1 * pbest_target + c2_r2 * gbest_target) / (C_gravity + 1e-16)

#                 # 构建二阶差分方程系数 (包含动态的 w)
#                 a1 = 1 + w_dynamic - C_gravity
#                 a2 = -w_dynamic

#                 # 计算自然偏移量 X_Q
#                 X_Q = a1 * (self.xs[i, d] - Q) + a2 * (self.xs_old[i, d] - Q)

#                 # RL 唯一干预点：强行缩放动态产生的自然位移
#                 new_xs[i, d] = Q + Conv_a * X_Q

#         # 越界处理
#         new_xs = np.clip(new_xs, self.pos_min, self.pos_max)

#         # 状态迭代
#         self.xs_old = self.xs.copy() 
#         self.xs = new_xs             

#         self.fits = self.fun(self.xs)
#         self.update_best()


##########固定参数策略+收敛性策略########### 
import numpy as np
from matAgent.testpso import TestpsoSwarm  

class FiftyDimCCPsoSwarm(TestpsoSwarm):
    optimizer_name = 'CCPSO_50D'
    action_space = 10
    obs_space = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = '50D_CC_PSO'
        # 记录上一代的位置 X(t-1)
        self.xs_old = self.xs.copy()

    def run_once(self, actions):
        if actions is None:
            actions = np.zeros(self.action_space * self.n_group)
            
        if self.show:
            print('{}|best fit:{}'.format(self.fe_num / self.fe_max, self.history_best_fit))

        new_xs = np.zeros_like(self.xs)

        # 固定w,c1,c2
        # c1 = 1.49445
        # c2 = 1.49445
        # 这里为了配合动态寻优，我们采用标准的线性衰减 [0.9 -> 0.4]
        # w_fixed = 0.9 - 0.5 * (self.fe_num / self.fe_max)

        # 与作者保持一致
        c1 = 2
        c2 = 2
        w = 0.5


        for i in range(self.n_part):
            # 提取动作
            action = actions[
                i % self.n_group * self.action_space:
                i % self.n_group * self.action_space + self.action_space
            ]

            # 【核心修正 2：遵循老师指导，RL 只输出并控制唯一参数 Conv_a】
            # 映射到 [0.1, 1.5] 的区间。
            # > 1 用于前期打破局部最优进行广度探索；< 1 用于后期精细收缩。
            Conv_a = action[0] * 0.7 + 0.8  
            

            for d in range(self.n_dim):
                # 为每个粒子的每个维度生成独立的均匀分布随机数
                r1 = np.random.rand()
                r2 = np.random.rand()

                pbest_target = self.p_best[i, d]
                gbest_target = self.g_best[d]

                c1_r1 = c1 * r1
                c2_r2 = c2 * r2

                C_gravity = c1_r1 + c2_r2  
                
                # 计算严谨的等效引力中心 Q
                Q = (c1_r1 * pbest_target + c2_r2 * gbest_target) / (C_gravity + 1e-16)

                # 构建二阶差分方程的系数
                a1 = 1 + w - C_gravity
                a2 = -w

                # 计算基于稳定底座的自然偏移量 X_Q
                X_Q = a1 * (self.xs[i, d] - Q) + a2 * (self.xs_old[i, d] - Q)

                # 唯一由强化学习干预的环节：缩放位移，实施收敛性控制
                new_xs[i, d] = Q + Conv_a * X_Q

        # 越界处理 (强行拉回搜索空间)
        new_xs = np.clip(new_xs, self.pos_min, self.pos_max)

        # 状态迭代：更新历史轨迹
        self.xs_old = self.xs.copy() 
        self.xs = new_xs             

        # 计算适应度并更新最优记录
        self.fits = self.fun(self.xs)
        self.update_best()       

########消融实验策略###########
# import numpy as np
# from matAgent.testpso import TestpsoSwarm  

# class FiftyDimCCPsoSwarm(TestpsoSwarm):
#     optimizer_name = 'CCPSO_50D'
#     action_space = 10
#     obs_space = 1

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.name = '50D_CC_PSO'
#         # 记录上一代的位置 X(t-1)，这是你的二阶方程必需的
#         self.xs_old = self.xs.copy()

#     def run_once(self, actions):
#         if actions is None:
#             actions = np.zeros(self.action_space * self.n_group)
#         if self.show:
#             print('{}|best fit:{}'.format(self.fe_num / self.fe_max, self.history_best_fit))

#         new_xs = np.zeros_like(self.xs)

#         for i in range(self.n_part):
#             # 完全保留原版的适应度差值计算 (用于 FDR)
#             fdr_deta_fitness = self.atom_history_best_fits[i] - self.atom_history_best_fits

#             # 提取原版 10 维动作
#             action = actions[
#                 i % self.n_group * self.action_space:
#                 i % self.n_group * self.action_space + self.action_space
#             ]

#             # ---------------------------------------------------------
#             # 严格保留原作者通过 RL 输出的所有参数映射
#             # ---------------------------------------------------------
#             w = action[7] * 0.4 + 0.5
#             r1 = action[1] * 1.5 + 1.5  # CLPSO 吸引力
#             r2 = action[2] * 1.5 + 1.5  # FDR 吸引力
#             r5 = action[5] * 1.5 + 1.5  # Gbest 吸引力
#             r6 = action[6] * 1.5 + 1.5  # Pbest 吸引力

#             # 原版的引力系数归一化逻辑
#             r = np.array([r1, r2, 0, 0, r5, r6])
#             r = r / (np.sum(r) + 1e-10) * (action[8] + 1) * 4
#             r1, r2, r3, r4, r5, r6 = r
            
#             # 【这里是你加塞的创新点】：挪用 action[0] 作为收敛控制系数 C
#             # 映射到 [0.1, 1.5]，允许外扩和收缩
#             Conv_a = action[0] * 0.7 + 0.8  

#             mutation_rate = (action[9] + 1) * 0.01

#             for d in range(self.n_dim):
#                 # ---------------------------------------------------------
#                 # 严格保留原作者的异构目标寻找机制 (控制变量法的关键)
#                 # ---------------------------------------------------------
#                 clpso_target = self.p_best[self.fid[i, d], d]

#                 xid = self.xs[i, d]
#                 distance = xid - self.p_best[:, d]
#                 distance[i] = np.inf
#                 fdr = fdr_deta_fitness / (distance + 1e-250)
#                 j_index = np.argmax(fdr)
#                 fdr_target = self.p_best[j_index, d]

#                 gbest_target = self.g_best[d]
#                 pbest_target = self.p_best[i, d]

#                 # ---------------------------------------------------------
#                 # 实施降维打击：将所有大杂烩目标，等效合并为一个物理重心 Q
#                 # ---------------------------------------------------------
#                 # 计算总引力 (即标准 PSO 里的 c1*r1 + c2*r2 的广义版本)
#                 C_gravity = r1 + r2 + r5 + r6

#                 # 计算所有目标的加权引力中心 Q
#                 Q = (r1 * clpso_target + r2 * fdr_target + r5 * gbest_target + r6 * pbest_target) / (C_gravity + 1e-16)

#                 # ---------------------------------------------------------
#                 # 你的核心消融机制：套用二阶差分收敛方程！
#                 # ---------------------------------------------------------
#                 # X(t) - Q = (1 + w - C_gravity)(X(t-1) - Q) - w(X(t-2) - Q)
#                 a1 = 1 + w - C_gravity
#                 a2 = -w

#                 # 此时的 X_Q 已经完全融合了作者所有的聪明才智(CLPSO/FDR/RL_w)
#                 X_Q = a1 * (self.xs[i, d] - Q) + a2 * (self.xs_old[i, d] - Q)

#                 # 利用你独有的收敛性策略进行最终降维控制
#                 new_xs[i, d] = Q + Conv_a * X_Q

#             # 严格保留原版的变异逻辑 (保证消融实验的严谨性)
#             if np.random.random() < mutation_rate * self.flag[i]:
#                 new_xs[i] = np.random.uniform(self.pos_min, self.pos_max, self.xs[i].shape)

#         # 越界处理
#         new_xs = np.clip(new_xs, self.pos_min, self.pos_max)

#         # 状态迭代：更新历史轨迹
#         self.xs_old = self.xs.copy() 
#         self.xs = new_xs             

#         self.fits = self.fun(self.xs)
#         self.update_best()