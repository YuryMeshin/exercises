import numpy as np
from utils import DiscreteRandomVariable
from numbers import Number

class BinomialModel():

    def __init__(self, s0: Number, u: Number, d: Number, p: Number, q: Number, rate: Number, periods: int=1):
        self.increment = DiscreteRandomVariable(np.array([u, d]), np.array([p, q]))
        self.periods = periods
        self.rate = rate
        self.s0 = s0
        self.risk_neutral = np.array([(1 + rate - d) / (u - d), (u - 1 - rate) / (u - d)])

    def pick(self) -> np.array:
        increments = np.array([self.s0])
        for _ in range(self.periods):
            x = self.increment.pick()
            increments = np.append(increments, [increments[-1] * x])
        return increments
    
    def european_option_evaluate(self, call: bool, strike: Number) -> float:
        if self.periods == 0:
            if call:
                payoff = lambda x: max(x - strike, 0)
            else:
                payoff = lambda x: max(strike - x, 0)
            return payoff(self.s0)
        else:
            u, d = self.increment.domain
            p, q = self.increment.probability
            bm_up = BinomialModel(self.s0 * u, u, d, p, q, self.rate, self.periods - 1)
            bm_down = BinomialModel(self.s0 * d, u, d, p, q, self.rate, self.periods - 1)
            return (self.risk_neutral[0] * bm_up.european_option_evaluate(call, strike) + \
                    self.risk_neutral[0] * bm_down.european_option_evaluate(call, strike)) / (1 + self.rate)
    