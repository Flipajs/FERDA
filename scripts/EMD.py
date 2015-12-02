from pulp import *

prob = LpProblem('EMD', LpMinimize)


regions_areas = []

flows = ['f11', 'f12', 'f21', 'f22']
costs = {
    'f11': 1,
    'f12': 2,
    'f21': 2,
    'f22': 1
}

flow_vars = LpVariable.dicts("Vars",flows,0)

sum_p = 150
sum_q = 155

prob += lpSum([costs[i]*flow_vars[i] for i in flows])
prob += lpSum(flow_vars['f11'] + flow_vars['f12']) <= 50
prob += lpSum(flow_vars['f21'] + flow_vars['f22']) <= 100
prob += lpSum(flow_vars['f11'] + flow_vars['f21']) <= 100
prob += lpSum(flow_vars['f12'] + flow_vars['f22']) <= 55
prob += lpSum([flow_vars[i] for i in flows]) <= min(sum_p, sum_q)
prob += lpSum([flow_vars[i] for i in flows]) >= min(sum_p, sum_q)


prob.solve()
print("Status:", LpStatus[prob.status])

for v in prob.variables():
    print(v.name, "=", v.varValue)