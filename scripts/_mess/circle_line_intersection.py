__author__ = 'filip@naiser.cz'

import math

m = [403, 387] #circle middle

c = [778, 338] #abscissa point
p = [780, 338] #abscissa point


r = 382 #circle radius
x = [0, 0]


a_ = c[0]**2 - 2*p[0]*c[0] + p[0]**2
a_ += c[1]**2 - 2*p[1]*c[1] + p[1]**2

b_ = 2*c[0]*p[0] - 2*c[0]*m[0] - 2*p[0]**2 + 2*p[0]*m[0]
b_ += 2*c[1]*p[1] - 2*c[1]*m[1] - 2*p[1]**2 + 2*p[1]*m[1]

c_ = p[0]**2 - 2*p[0]*m[0] + m[0]**2
c_ += p[1]**2 - 2*p[1]*m[1] + m[1]**2
c_ -= r**2

d_ = math.sqrt(b_**2 - 4*a_*c_)
alpha1 = (-b_ + d_) / (2*a_)
alpha2 = (-b_ - d_) / (2*a_)

alpha = 0
if 0 <= alpha1 <= 1:
    alpha = alpha1
elif 0 <= alpha2 <= 1:
    alpha = alpha2

x[0] = alpha * c[0] + (1 - alpha) * p[0]
x[1] = alpha * c[1] + (1 - alpha) * p[1]

print x