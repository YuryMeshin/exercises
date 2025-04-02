from typing import Optional, Union, Self
from numbers import Number
from itertools import product


class DiscreteRandomVariable():

    def __init__(self, domain: list[float], precision: int=0, \
                 weights: Optional[list[float]]=None, name: str='X'):
        self.name = name
        self.precision = precision
        assert len(domain) == len(set([round(x, precision) for x in domain])), \
            'Given domain values must be unique'
        self.domain = [round(x, self.precision) for x in sorted(domain)]
        if weights:
            assert len(weights) == len(self.domain), \
                f"Given domain size {len(domain)} doesn't match weights size {len(weights)}"
            assert (w_sum := sum(weights)) > 0, f"Total weight of weights given must be positive"
            self.probability = [w / w_sum for w in weights]
        else:
            self.probability = [1 / len(domain) for _ in domain]
    
    def __repr__(self, precision: int=2) -> str:
        header = f'Discrete Random Variable {self.name}:\n'
        template = f'\tP({self.name}=' + '{:.' + str(self.precision) + 'f})={:.' + str(precision) + 'f}\n'
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
    
    def __add__(self, other: Union[Self, Number]) -> Self:
        if isinstance(other, Number):
            return DiscreteRandomVariable([d + other for d in self.domain], self.precision, \
                                            self.probability, self.name)
        elif isinstance(other, DiscreteRandomVariable):
            n = len(self.domain)
            m = len(other.domain)
            precision = max(self.precision, other.precision)
            values = {}
            for i, j in product(range(n), range(m)):
                new_val = round(self.domain[i] + other.domain[j], precision)
                new_weight = self.probability[i] * other.probability[j]
                values[new_val] = values.get(new_val, 0) + new_weight
            return DiscreteRandomVariable(list(values.keys()), precision, \
                                            values.values(), '+'.join([self.name, other.name]))
    
    def __sub__(self, other: Union[Self, Number]) -> Self:
        if isinstance(other, Number):
            return DiscreteRandomVariable([d - other for d in self.domain], self.precision, \
                                            self.probability, self.name)
        elif isinstance(other, DiscreteRandomVariable):
            n = len(self.domain)
            m = len(other.domain)
            precision = max(self.precision, other.precision)
            values = {}
            for i, j in product(range(n), range(m)):
                new_val = round(self.domain[i] - other.domain[j], precision)
                new_weight = self.probability[i] * other.probability[j]
                values[new_val] = values.get(new_val, 0) + new_weight
            return DiscreteRandomVariable(list(values.keys()), precision, \
                                            values.values(), '-'.join([self.name, other.name]))
    
    def __mul__(self, other: Union[Self, Number]) -> Self:
        if isinstance(other, Number):
            return DiscreteRandomVariable([d * other for d in self.domain], self.precision, \
                                            self.probability, self.name)
        elif isinstance(other, DiscreteRandomVariable):
            n = len(self.domain)
            m = len(other.domain)
            precision = max(self.precision, other.precision)
            values = {}
            for i, j in product(range(n), range(m)):
                new_val = round(self.domain[i] * other.domain[j], precision)
                new_weight = self.probability[i] * other.probability[j]
                values[new_val] = values.get(new_val, 0) + new_weight
            return DiscreteRandomVariable(list(values.keys()), precision, \
                                            values.values(), '*'.join([self.name, other.name]))
    
    def __truediv__(self, other: Union[Self, Number]) -> Self:
        if isinstance(other, Number):
            if other == 0:
                raise ZeroDivisionError('Given value of the divisor is equal to 0')
            return DiscreteRandomVariable([d / other for d in self.domain], self.precision, \
                                            self.probability, self.name)
        elif isinstance(other, DiscreteRandomVariable):
            if 0 in other.domain:
                raise ZeroDivisionError(f'Given random variable {other.name} has value of 0 in its domain')
            n = len(self.domain)
            m = len(other.domain)
            precision = 10
            values = {}
            for i, j in product(range(n), range(m)):
                new_val = round(self.domain[i] / other.domain[j], precision)
                new_weight = self.probability[i] * other.probability[j]
                values[new_val] = values.get(new_val, 0) + new_weight
            return DiscreteRandomVariable(list(values.keys()), precision, \
                                            values.values(), '/'.join([self.name, other.name]))
