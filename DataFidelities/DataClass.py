'''
Abstract base class for specifying data-fidelity objects.
Jianxing Liao, CIG, WUSTL, 2018
Based on MATLAB code by U. S. Kamilov, CIG, WUSTL, 2017
'''
from abc import ABC, abstractmethod

class DataClass(ABC):
    @abstractmethod
    def size(self):
        pass
    @abstractmethod
    def evl(self,x):
        pass
    @abstractmethod
    def grad(self,x):
        pass
