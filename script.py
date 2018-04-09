# import some dependencies
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set()
except ImportError:
    pass

import torch
from torch.autograd import Variable

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

import numpy as np

torch.manual_seed(101)

# -----------Example--------------- 

def scale(guess):
    # The prior over weight encodes our uncertainty about our guess
    weight = pyro.sample("weight", dist.normal, guess, Variable(torch.ones(1)))
    # This encodes our belief about the noisiness of the scale:
    # the measurement fluctuates around the true weight
    return pyro.sample("measurement", dist.normal, weight, Variable(torch.Tensor([0.75])))

posterior = pyro.infer.Importance(scale, num_samples=100)

guess = Variable(torch.Tensor([8.5]))

marginal = pyro.infer.Marginal(posterior)

print(marginal(guess))

plt.hist([marginal(guess).data[0] for _ in range(100)], range=(5.0, 12.0))
plt.title("P(measurement | guess)")
plt.xlabel("weight")
plt.ylabel("#")
plt.show()

# ------------Normal distribution----------------

samples = []

for i in range(1000):
    mu = Variable(torch.zeros(1))
    sigma = Variable(torch.ones(1))
    samples.append((dist.normal(mu, sigma)).data[0])

plt.hist(samples, range=(-8.0, 8.0))
plt.title("P(sample)")
plt.xlabel("value")
plt.ylabel("#")
plt.show()

# --------------Exponential distribution--------------

samples = []

for i in range(1000):
    _lambda = Variable(torch.Tensor([0.5]))
    samples.append((dist.exponential(_lambda)).data[0])

plt.hist(samples, range=(0.0, 8.0))
plt.title("P(sample)")
plt.xlabel("value")
plt.ylabel("#")
plt.show()

# ------------Normal distribution with condition----------------

samples = []

def normal(guess):
    mu = pyro.sample("mu", dist.normal, guess, Variable(torch.Tensor([1.0])))
    # sigma = pyro.sample("sigma", dist.normal, Variable(torch.Tensor([1.0]), Variable(torch.Tensor([0.5]))))
    return pyro.sample("sample", dist.normal, mu, Variable(torch.Tensor([1.0])))

guess = Variable(torch.Tensor([0.0]))
sample = Variable(torch.Tensor([1.5]))

conditioned_normal = pyro.condition(normal, data={"sample": sample})

marginal = pyro.infer.Marginal(
    pyro.infer.Importance(conditioned_normal, num_samples=100), sites=["mu"])

# The marginal distribution concentrates around the data
print(marginal(guess))
plt.hist([marginal(guess)["mu"].data[0] for _ in range(100)], range=(-4.0, 4.0))
plt.title("P(sample | mu)")
plt.xlabel("value")
plt.ylabel("#")
plt.show()

# ------------Normal distribution with condition----------------

samples = []

def normal(guess):
    mu = pyro.sample("mu", dist.normal, guess, Variable(torch.Tensor([1.0])))
    sig = pyro.sample("sig", dist.normal, Variable(torch.Tensor([1.0]), Variable(torch.Tensor([0.5]))))
    return pyro.sample("sample", dist.normal, mu, sig)

guess = Variable(torch.Tensor([0.0]))

conditioned_normal = pyro.condition(
    normal, data={"mu": Variable(torch.Tensor([1.0])),
                  "sig": Variable(torch.Tensor([1.0])),
                  "sample": Variable(torch.Tensor([1.0]))})

marginal = pyro.infer.Marginal(
    pyro.infer.Importance(conditioned_normal, num_samples=100), sites=["sig"])

# The marginal distribution concentrates around the data
print(marginal(guess))
plt.hist([marginal(guess)["mu"].data[0] for _ in range(100)], range=(-4.0, 4.0))
plt.title("P(sample | mu)")
plt.xlabel("value")
plt.ylabel("#")
plt.show()

