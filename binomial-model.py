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
        self.increment_risk_neutral = DiscreteRandomVariable(np.array([u, d]), self.risk_neutral)

    def pick(self) -> np.array:
        increments = np.array([self.s0])
        for _ in range(self.periods):
            x = self.increment.pick()
            increments = np.append(increments, [increments[-1] * x])
        return increments

    def european_option_evaluate(self, call: bool, strike: Number) -> float:
        if call:
            payoff = lambda x: max(x - strike, 0)
        else:
            payoff = lambda x: max(strike - x, 0)
        expectation = DiscreteRandomVariable([self.s0], [1])
        for _ in range(self.periods):
            expectation *= self.increment_risk_neutral
        return expectation.transform(payoff).expectation() / ((1 + self.rate) ** self.periods)
    
    def delta(self, call: bool, strike: Number) -> float:
        assert self.periods > 0, 'Option has already been expired'
        u, d = self.increment.domain[1], self.increment.domain[0]
        p, q = self.increment.probability[1], self.increment.probability[0]
        option_up = type(self)(self.s0 * u, u, d, p, q, self.periods - 1).european_option_evaluate(call, strike)
        option_down = type(self)(self.s0 * d, u, d, p, q, self.periods - 1).european_option_evaluate(call, strike)
        return (option_up - option_down) / (u - d) / self.s0
