
# coding: utf-8

# In[11]:


'''
Find the global maximum for function: f(x) =  -3*(x-30)**2*sin(x)
'''
#画图
import matplotlib.pyplot as plt
import numpy as np
import random
x = np.arange(0, 15, 0.001)
y = -3*(x-30)**2*np.sin(x)
plt.title("y = -3*(x-30)**2*sin(x)")
plt.plot(x, y)
plt.show()


# In[2]:


#导入种群和内置算子的相关类
from math import sin, cos
from gaft import GAEngine
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitMutation


# In[3]:


# 用于编写分析插件的接口类 
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis


# In[4]:


# 内置的存档适应度函数的分析类 
from gaft.analysis.fitness_store import FitnessStore


# In[5]:


# 定义种群 
indv_template = BinaryIndividual(ranges=[(0, 15)], eps=0.001)
population = Population(indv_template=indv_template, size=30).init()


# In[6]:


# 创建遗传算子
selection = TournamentSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)


# In[7]:


# 创建遗传算法引擎, 分析插件和适应度函数可以以参数的形式传入引擎中 
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[FitnessStore])


# In[8]:


#定义适应度函数
@engine.fitness_register
def fitness(indv):
    x, = indv.solution
    return  -3*(x-30)**2*sin(x)


# In[9]:


# Define on-the-fly analysis.
@engine.analysis_register
class ConsoleOutputAnalysis(OnTheFlyAnalysis):
    interval = 1
    master_only = True

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        msg = 'Generation: {}, best fitness: {:.3f}'.format(g, engine.ori_fmax)
        self.logger.info(msg)

    def finalize(self, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x = best_indv.solution
        y = engine.ori_fmax
        msg = 'Optimal solution: ({}, {})'.format(x, y)
        self.logger.info(msg)


# In[10]:


if '__main__' == __name__:
    # Run the GA engine.
    engine.run(ng=100)

