from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.linear_model import Ridge as lg

import numpy as np

import pickle as pk

import math

import time

GATEWAYS = ["11c85b", "1f1bf9", "62f77a", "a54379", "bacef8", "f6c9fe"]

global g_users, g_average_model, g_gw_model

# Setting-up simulated annealing parameters
N_SA = 2
K_SA = 20
INITIAL_SCHEDULE = 0.2*np.ones(K_SA)
SCHEDULE_SA = np.zeros(N_SA*K_SA)
for i in range(N_SA):
    SCHEDULE_SA[i*K_SA:(i+1)*K_SA] = INITIAL_SCHEDULE * 0.5**(i)


# Defining the decision variable range
T_LOW = 22.5
T_HIGH = 27
CONTROL_DECISIONS = np.arange(T_LOW, T_HIGH+0.5, 0.5)

# Setting-up the parameters for power prediction models
T_SLOT = 15 * 60
Q = 3
R = Q * 1.2 * 1.006
A = 1.0/(60*5)
Q_INT = Q/3
A_IO = R/2
W_MAX = 10

# setting-up cost parameters
COST_POWER = 1
PENALTY_PMV = 350
PENALTY_FACTOR = 3

M = 4
MAX_REJECT = 30


#class GatewayError(Exception):
#    def __init__(self, msg):
#        self.__message = msg
#    def __str__(self):
#        return repr(self.__message)

def K1(d, out_temp):
    return (R * d + Q_INT + A_IO * out_temp) / (R + A_IO)

def K2(d, out_temp):
  return (Q_INT + A_IO * out_temp - W_MAX)/(A_IO)


def T12(d, in_temp, out_temp):

    k2 = K2(d, out_temp)

    # prohibiting taking a negative number in the logarithm
    if (d + W_MAX / R - k2) <=0 or ( in_temp - k2 <= 0): #NOTE: in_temp-k2 can be negative
        return T_SLOT * 100
    else:
        return - math.log((d + W_MAX / R - k2) / ( in_temp - k2))/(A * A_IO)


# power consumtion predictor (theoretical function)
def h_power(d, sensor_values):
    in_temp = sensor_values[0]
    out_temp = sensor_values[2]
    k1 = K1(d, out_temp)
    k2 = K2(d, out_temp)
    t12 = T12(d, in_temp, out_temp)

    ret = 0
    if (in_temp <= d):
        ret = 0
    elif ( in_temp <= d + W_MAX/R):
        ret = R * (k1 + (in_temp - k1)/T_SLOT * (1 - math.exp(-A * (R + A_IO) * T_SLOT))/ (A * (R + A_IO))-d)
    elif (T_SLOT < t12):
        ret = W_MAX
    else:
        ret = W_MAX * t12 / T_SLOT + R * ( k1 * (T_SLOT - t12) / T_SLOT
                        +  (in_temp - k1)/T_SLOT * (math.exp(-A * (R + A_IO) * t12) - math.exp(-A * (R + A_IO) * T_SLOT) )/ (A * (R + A_IO))
                        - d * (T_SLOT - t12) / T_SLOT )

    return ret

# thremal condition predictor (theoretical function)
def f_TC(d, sensor_values):
    in_temp = sensor_values[0]
    out_temp = sensor_values[2]
    k1 = K1(d, out_temp)
    k2 = K2(d, out_temp)
    t12 = T12(d, in_temp, out_temp)

    next_in_temp = 0
    if ( in_temp < d + W_MAX/R):
        next_in_temp =  k1 + (in_temp - k1)/T_SLOT * (1 - math.exp(-A * (R + A_IO) * T_SLOT))/ (A * (R + A_IO))
    elif (T_SLOT < t12):
        next_in_temp = k2 + (in_temp - k2)/T_SLOT * (1 - math.exp(-A * A_IO * T_SLOT))/ (A * A_IO)
    else:
        next_in_temp = k2 * t12 / T_SLOT + (in_temp - k2) / T_SLOT * (1 - math.exp(-A * A_IO * t12))/ (A * A_IO) \
                       + k1 * (T_SLOT - t12) / T_SLOT \
                       + (in_temp - k1)/T_SLOT * (math.exp(-A * (R + A_IO) * t12) \
                                                  - math.exp(-A * (R + A_IO) * T_SLOT) )/ (A * (R + A_IO))
    sv = np.copy(sensor_values)
    sv[0] = next_in_temp
    sv[4] = next_in_temp - in_temp
    return sv


def predict_PMV(sensor_values, user_info):
    #check if any user is in the gateway's user list
    d = len(sensor_values)

    user_list = []
    stranger_list = []
    for i, (id, age) in enumerate(user_info):
        is_stranger = True
        for j, user in enumerate(g_users):
            if user == id:
                user_list.append(j)
                is_stranger = False
                break
        if is_stranger:
            stranger_list.append(age)
    rets = np.zeros(len(user_info))
    if len(user_list) >0:
        inputs = np.zeros([len(user_list), d+len(g_users)])
        for i,idx in enumerate(user_list):
            inputs[i, :d] = sensor_values
            inputs[i, d+idx] = 1
        rets[:len(user_list)] = g_gw_model.predict(inputs)

    if len(stranger_list) >0 :
        inputs = np.zeros([len(stranger_list), d + 1])
        for i, age in enumerate(stranger_list):
            inputs[i, 1:] = sensor_values
            inputs[i, 0] = age
        rets[len(user_list):] = g_average_model.predict(inputs)
    return np.mean(rets)

# calculating cost for a unit slot given sensor data
def cost_slot(d, sensor_values, user_info):
    power = h_power(d, sensor_values)
    next_sensor_values = f_TC(d, sensor_values)
    PMV = predict_PMV(next_sensor_values, user_info)
    return COST_POWER * power + PENALTY_PMV * abs(PMV)**PENALTY_FACTOR




# function that generate an initial control sequence through the greedy algorithm
def get_initial_solution(sensor_values, ids_present):
    d_initial  = np.zeros(M, dtype = int)
    cost_initial_slot = np.zeros(M, dtype = float)
    sv = sensor_values

    for k in range(M):
        #  get costs during the k_th next time slot for various control decision values
        min_d = -100
        min_cost = 10**10

        for d in CONTROL_DECISIONS:
            cost = cost_slot(d, sv, ids_present)
            if cost < min_cost:
                min_d = d
                min_cost = cost

        # finding the control decision under which the cost is the minimum among the control decision values
        d_initial[k] = min_d
        # setting the initial cost at the k_the next time slot
        cost_initial_slot[k] = min_cost
        # predict the thermal condition at the (k+1)_th time slot given the decision
        sv = f_TC(d_initial[k], sv)
    return d_initial, cost_initial_slot, np.sum(cost_initial_slot)

def search_optimal_solution(sensor_values, user_info):

    # setting up the initial greedy solution as the current solution
    current_decisions, current_cost_slot, current_cost = get_initial_solution(sensor_values, user_info)
    #print [current_cost, current_decisions, current_cost_slot]

    # list for tracing the history of updating the best solution
    best_costs = [current_cost]
    best_decisions = [current_decisions]

    # Setting up the simulated annealing schedule
    schedule_sa = SCHEDULE_SA * current_cost
    for schedule in schedule_sa:
        # clearing the reject count
        reject_count = 0

        # mutating the current solution on a random basis
        mutation_decisions = current_decisions + np.random.choice(np.array([-0.5, 0, 0.5]), M)

        for k in range(M):
            if mutation_decisions[k] < T_LOW:
                mutation_decisions[k] = T_LOW
            if mutation_decisions[k] > T_HIGH:
                mutation_decisions[k] = T_HIGH


        # setting up the sensor values for the current time slot
        sv = sensor_values
        mutation_cost_slot = np.zeros(M, dtype = float)
        for k in range(M):
            # calculating at the next k_th time slot for the mutation solution
            mutation_cost_slot[k] = cost_slot(mutation_decisions[k], sv, user_info)
            # updating the thermal condition at the next (k+1)_th slot under the k_th slot decision
            sv = f_TC(mutation_decisions[k], sv)

        # Summing the total cost over time slots for the mutation solution
        mutation_cost = sum(mutation_cost_slot)

        # If the mutation solution is better than the current solution, then update the current solution
        if mutation_cost <= current_cost:
            current_decisions = mutation_decisions
            current_cost_slot = mutation_cost_slot
            current_cost = mutation_cost

            # If the mutation solution is better than the best solution, then update the best solution

            if best_costs[-1] > current_cost:
                best_costs.append(current_cost)
                best_decisions.append(current_decisions)
                #print [current_cost, current_decisions, current_cost_slot]

            reject_count = 0
        # If the mutation solution is worse than the curren solution,
        # then we accept the mutation solution as the current solution with a certain acceptance probability
        else:
            prob_accept = math.exp((current_cost - mutation_cost) / schedule)

            if np.random.uniform() <= prob_accept:
                current_decisions = mutation_decisions
                current_cost_slot = mutation_cost_slot
                current_cost = mutation_cost
                reject_count = 0
            else:
                reject_count += 1
                if reject_count > MAX_REJECT:
                    break

    return (best_costs, best_decisions)

def read_users(file_name):
    """"
    read list of users from gateway sensor data file
    """
    f = open(file_name, "r")
    header = f.readline()
    f.close()
    cols = [x.strip("\"\n") for x in header.split('\t')]
    return cols[12:]

def train_rf(gateway_name=None):
    """"
    train random forest model for a gateway
    or train random forest model using data from all gateways
    :param gateway_name name of the gateway
    :returns random forest model and list of users of the gateway
    """
    users = None
    if gateway_name and len(gateway_name) > 0:
        file_name = "data/" + gateway_name+".txt"
        users = read_users(file_name)

        data = np.genfromtxt(file_name, delimiter='\t',skip_header=True)
        out_file = "models/" + gateway_name + "_model.pk"
    else:
        data = np.genfromtxt('data/all.txt', delimiter='\t', skip_header=True)
        out_file = "models/all_model.pk"

    model = rf(n_estimators=10, max_features=6)
    if users:
        model.fit(data[:, 2:], data[:, 0])
        print "R2(%s)=%f" % (gateway_name, model.score(data[:, 2:], data[:, 0]))
        pk.dump([model, users], open(out_file, "w"))
    else:
        model.fit(data[:, 1:], data[:, 0])
        print "R2(all)=%f" % model.score(data[:, 1:], data[:, 0])
        pk.dump(model, open(out_file, "w"))


def train_lg(gateway_name=None):
    """"
    train random forest model for a gateway
    or train random forest model using data from all gateways
    :param gateway_name name of the gateway
    :returns random forest model and list of users of the gateway
    """
    users = None
    if gateway_name and len(gateway_name) > 0:
        file_name = "data/" + gateway_name+".txt"
        users = read_users(file_name)

        data = np.genfromtxt(file_name, delimiter='\t',skip_header=True)
        out_file = "models/" + gateway_name + "_model.pk"
    else:
        data = np.genfromtxt('data/all.txt', delimiter='\t', skip_header=True)
        out_file = "models/all_model.pk"

    model = lg()
    if users:
        model.fit(data[:, 2:], data[:, 0])
        print "R2(%s)=%f" % (gateway_name, model.score(data[:, 2:], data[:, 0]))
        pk.dump([model, users], open(out_file, "w"))
    else:
        model.fit(data[:, 1:], data[:, 0])
        print "R2(all)=%f" % model.score(data[:, 1:], data[:, 0])
        pk.dump(model, open(out_file, "w"))


def load_model(gateway_name=None):
    """"
    load a trained model from file (pickle)
    :param gateway_name name of the gateway (empty for the average model
    :return loaded model (and list of users of the gateway)
    """
    if gateway_name and len(gateway_name) > 0:
        model = pk.load(open("models/" + gateway_name + "_model.pk", "r"))
    else:
        model = pk.load(open("models/all_model.pk", "r"))
    return model

def check_params():
    """"
    just to print out some parameter values for checking with R code
    """
    print INITIAL_SCHEDULE
    print SCHEDULE_SA[:10]
    print CONTROL_DECISIONS

    print "T_SLOT= %f" % T_SLOT
    print "R=%f" % R
    print "A=%f" % A
    print "A_IO=%f" % A_IO
    print "Q_INT=%d" % Q_INT

def train():
    #training and save models
    print "start training"
    start = time.time()
    for gw in GATEWAYS:
        train_rf(gw)
    train_rf()
    end = time.time()
    print"completed in %f secs" % (end - start)

def load():
    global g_average_model, g_gw_model, g_users
    print "start loading models"
    start = time.time()
    g_average_model = load_model()
    g_gw_model, g_users = load_model(GATEWAYS[1])
    end = time.time()
    print"completed loading in %f secs" % (end - start)

def test_prediction():
    print "start predicting"
    start = time.time()

    sensor_values = np.array([26.32667, 68.62, 29.3, 56, 0, -0.5733333, 0.7857129, -2.142857, 35, 10])
    user_info = [('c232', 39), ('d236', 41), ('s2218',34), ('y2622',33), ('abc123', 45)]
    pmv = predict_PMV(sensor_values, user_info)
    end = time.time()
    print"completed prediction in %f secs" % (end - start)
    print "predicted PMV: %f" % pmv
    #print costs


def test_optimization():
    print "start optimization"
    start = time.time()

    sensor_values = np.array([26.32667, 68.62, 29.3, 56, 0, -0.5733333, 0.7857129, -2.142857, 35, 10])
    #user_info = [('c232', 39), ('d236', 41), ('j2610', 42), ('k2515', 30), ('s2218',34), ('y2622',33)] #userlist in gateway 1f1bf9
    #user_info = [('c232', 39), ('d236', 41), ('s2218',34), ('y2622',33), ('abc123', 45)]
    user_info = [('c232', 39), ('d236', 41), ('s2218', 34), ('y2622', 33)]


    costs, decisions = search_optimal_solution(sensor_values, user_info)
    end = time.time()
    print"completed prediction in %f secs" % (end - start)
    min_idx = np.argmin(costs)
    print "best decision sequence: ", decisions[min_idx], " with cost: ", costs[min_idx]
    #print costs

def main():

    #train models
    train()

    #2. load saved models
    load()

    #3. make predictions

    test_prediction()
    test_optimization()



if __name__ == '__main__':
    main()