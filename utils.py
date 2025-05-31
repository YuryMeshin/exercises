import numpy as np
from typing import Optional, Union, Self, Callable, Iterable
from numbers import Number
from itertools import product
import random


class DiscreteRandomVariable():

    def __init__(self, domain: Iterable[Number], weights: Optional[Iterable[Number]]=None, name: str='X'):
        assert isinstance(name, str), f'Only string names are accepted'
        if weights is None:
            data = [[d, 1 / len(domain)] for i, d in enumerate(domain)]
        else:
            assert len(weights) == len(domain), \
                f"Given domain size {len(domain)} doesn't match weights size {len(weights)}"
            assert (w_sum := sum(weights)) > 0, f"Total weight of weights given must be positive"
            weights = np.array(weights, dtype=float)
            weights /= w_sum
            data = [[d, weights[i]] for i, d in enumerate(domain)]
        data.sort()
        self.name = name
        self.domain = np.array([d[0] for d in data])
        self.probability = np.array([d[1] for d in data])
    
    def rename(self, name: str):
        assert isinstance(name, str), f'Only string names are accepted'
        self.name = name
    
    def __repr__(self) -> str:
        header = f'Discrete Random Variable {self.name}:\n'
        template = f'\tP({self.name}=' + '{:.5f})={:.5f}\n'
        chunks = [header]
        if len(self.domain) > 5:
            for i in range(2):
                chunks.append(template.format(self.domain[i], self.probability[i]))
            chunks.append('\t...\n')
            for i in range(-2, 0):
                chunks.append(template.format(self.domain[i], self.probability[i]))
        else:
            for i in range(len(self.domain)):
                chunks.append(template.format(self.domain[i], self.probability[i]))
        return ''.join(chunks)
    
    def copy(self, name: Optional[str]=None) -> 'DiscreteRandomVariable':
        assert isinstance(name, Optional[str]), \
            f'Unsupported name type {type(name)} given, only string or NoneType are supported'
        return type(self)(self.domain.copy(), self.probability.copy(), name if name else self.name)
    
    def __eq__(self, other: Union[Self, Number]) -> bool:
        if isinstance(other, type(self)):
            if self.domain.shape[0] != other.domain.shape[0]:
                return False
            if max(abs(self.domain - other.domain)) > 1e-10:
                return False
            if max(abs(self.probability - other.probability)) > 1e-10:
                return False
            return True
        elif isinstance(other, Number):
            if len(self.domain) == 1:
                return abs(self.domain[0] - other) < 1e-10
            else:
                return False
        return NotImplemented
    
    def __ne__(self, other: Union[Self, Number]) -> bool:
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result

    def __add__(self, other: Union[Self, Number]) -> 'DiscreteRandomVariable':
        if isinstance(other, Number):
            return DiscreteRandomVariable(self.domain + other, self.probability, \
                                          f'{self.name}+{other:.5f}')
        elif isinstance(other, type(self)):
            n, m = len(self.domain), len(other.domain)
            values = {}
            for i, j in product(range(n), range(m)):
                new_val = round(self.domain[i] + other.domain[j], 5)
                new_weight = self.probability[i] * other.probability[j]
                values[new_val] = values.get(new_val, 0) + new_weight
            return DiscreteRandomVariable(list(values.keys()), list(values.values()), \
                                          '+'.join([self.name, other.name]))
    
    def __sub__(self, other: Union[Self, Number]) -> 'DiscreteRandomVariable':
        if isinstance(other, Number):
            return DiscreteRandomVariable(self.domain - other, self.probability, \
                                          f'{self.name}-{other:.5f}')
        elif isinstance(other, type(self)):
            n, m = len(self.domain), len(other.domain)
            values = {}
            for i, j in product(range(n), range(m)):
                new_val = round(self.domain[i] - other.domain[j], 5)
                new_weight = self.probability[i] * other.probability[j]
                values[new_val] = values.get(new_val, 0) + new_weight
            return DiscreteRandomVariable(list(values.keys()), list(values.values()), \
                                          '-'.join([self.name, other.name]))
    
    def __mul__(self, other: Union[Self, Number]) -> 'DiscreteRandomVariable':
        if isinstance(other, Number):
            if (abs(other) < 1e-10):
                return DiscreteRandomVariable([0], [1], f'{self.name}*0')
            else:
                return DiscreteRandomVariable(self.domain * other, self.probability, \
                                              f'{self.name}*{other:.5f}')
        elif isinstance(other, type(self)):
            n, m = len(self.domain), len(other.domain)
            values = {}
            for i, j in product(range(n), range(m)):
                new_val = round(self.domain[i] * other.domain[j], 5)
                new_weight = self.probability[i] * other.probability[j]
                values[new_val] = values.get(new_val, 0) + new_weight
            return DiscreteRandomVariable(list(values.keys()), list(values.values()), \
                                          '*'.join([self.name, other.name]))
    
    def transform(self, func: Optional[Callable]=lambda x: x, precision: Optional[int]=None) -> 'DiscreteRandomVariable':
        values = {}
        for i, d in enumerate(self.domain):
            values[func(d)] = values.get(func(d), 0) + self.probability[i]
        return DiscreteRandomVariable(list(values.keys()), list(values.values()), self.name)
        
    def expectation(self, func: Optional[Callable]=lambda x: x, precision: Optional[int]=None) -> float:
        result = sum([func(self.domain[i]) * self.probability[i] for i in range(len(self.domain))])
        if precision:
            return round(result, precision)
        else:
            return result
    
    def dispersion(self, precision: Optional[int]=None) -> float:
        _exp = self.expectation()
        func = lambda x: (x - _exp) ** 2
        return self.expectation(func, precision)

    def pick(self) -> float:
        threshold = random.uniform(0, 1)
        i, w = 0, self.probability[0]
        while (threshold > w):
            i += 1
            w += self.probability[i]
        return self.domain[i]
