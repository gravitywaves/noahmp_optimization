# src/optimization/sceua.py
import numpy as np
from typing import List, Tuple, Callable
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

@dataclass
class SCEUAConfig:
    """SCE-UA算法配置"""
    num_complexes: int              # 复形数量
    points_per_complex: int         # 每个复形中的点数
    min_complexes: int = 2          # 最小复形数
    max_iterations: int = 10000     # 最大迭代次数
    max_evaluations: int = 1000000  # 最大评估次数
    num_evolution_steps: int = None # 每个复形的进化步数
    convergence_tol: float = 1e-5   # 收敛阈值
    random_seed: int = None         # 随机种子
    parallel: bool = False          # 是否并行计算
    num_workers: int = 4            # 并行工作进程数

class SCEUA:
    """SCE-UA优化算法实现"""

    def __init__(self,
                 num_params: int,
                 bounds: List[Tuple[float, float]],
                 config: SCEUAConfig = None):
        """
        初始化SCE-UA优化器

        Args:
            num_params: 参数个数
            bounds: 参数边界
            config: SCE-UA配置
        """
        self.num_params = num_params
        self.bounds = np.array(bounds)

        # 设置默认配置
        if config is None:
            config = self._get_default_config()
        self.config = config

        # 初始化算法参数
        self.points_per_complex = max(2 * num_params + 1, config.points_per_complex)
        self.num_points_in_subcomplex = num_params + 1
        self.population_size = self.config.num_complexes * self.points_per_complex

        # 设置随机种子
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        # 初始化评估计数器
        self.num_evaluations = 0

        # 设置日志
        self.logger = logging.getLogger('SCE-UA')

    def optimize(self, objective_function: Callable) -> Tuple[np.ndarray, float]:
        """运行优化"""
        self.obj_func = objective_function

        # 初始化种群
        population, fitness = self._initialize_population()
        best_fitness_history = [np.min(fitness)]

        # 主循环
        iteration = 0
        while not self._check_convergence(best_fitness_history):
            # 将种群分成复形
            complexes = self._partition_into_complexes(population, fitness)

            # 演化每个复形
            if self.config.parallel:
                with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                    evolved_complexes = list(executor.map(self._evolve_complex, complexes))
            else:
                evolved_complexes = [self._evolve_complex(complex_) for complex_ in complexes]

            # 重新组合种群
            population = self._shuffle_complexes(evolved_complexes)
            fitness = np.array([self._evaluate(p) for p in population])

            # 更新历史记录
            best_fitness_history.append(np.min(fitness))

            # 记录进度
            if iteration % 100 == 0:
                self.logger.info(f"Iteration {iteration}, Best fitness: {best_fitness_history[-1]}")

            iteration += 1

            # 检查是否需要减少复形数量
            if len(complexes) > self.config.min_complexes:
                if self._should_reduce_complexes(best_fitness_history):
                    self.config.num_complexes -= 1
                    self.population_size = self.config.num_complexes * self.points_per_complex
                    # 保留最好的点
                    sort_idx = np.argsort(fitness)
                    population = population[sort_idx[:self.population_size]]
                    fitness = fitness[sort_idx[:self.population_size]]

        # 返回最优解
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def _competitive_complex_evolution(self, subcomplex: np.ndarray) -> np.ndarray:
        """竞争复形演化(CCE)算法"""
        # 计算质心(不包括最差点)
        centroid = np.mean(subcomplex[:-1], axis=0)

        # 反射
        alpha = 1.0
        reflected_point = centroid + alpha * (centroid - subcomplex[-1])
        reflected_point = self._ensure_bounds(reflected_point)
        reflected_fitness = self._evaluate(reflected_point)

        if reflected_fitness < self._evaluate(subcomplex[-1]):
            # 如果反射成功，尝试扩展
            gamma = 2.0
            expanded_point = centroid + gamma * (reflected_point - centroid)
            expanded_point = self._ensure_bounds(expanded_point)
            expanded_fitness = self._evaluate(expanded_point)

            if expanded_fitness < reflected_fitness:
                return expanded_point
            else:
                return reflected_point
        else:
            # 如果反射失败，进行收缩
            beta = 0.5
            contracted_point = centroid + beta * (subcomplex[-1] - centroid)
            contracted_point = self._ensure_bounds(contracted_point)
            contracted_fitness = self._evaluate(contracted_point)

            if contracted_fitness < self._evaluate(subcomplex[-1]):
                return contracted_point
            else:
                # 如果收缩也失败，随机生成新点
                return self._generate_random_point()

    # ... [其他SCE-UA类方法] ...
