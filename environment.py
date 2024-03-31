import numpy as np

class V2GEnvironment:
    def __init__(self, max_charge_rate, max_discharge_rate, battery_capacity, electricity_prices, gamma=0.99):
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.battery_capacity = battery_capacity
        self.electricity_prices = electricity_prices
        self.gamma = gamma
        self.state = self.reset() #initial configuration state
    
    #state contains whether ev is at home or not, current battery level and past 24 hrprices
    def reset(self):
        # Initialize the state based on the problem description
        ut = 1 #ut is set to 1 to indicate the EV is at home
        Et = self.battery_capacity / 2 #battery energy starts at half capacity
        #past_prices = np.zeros(24) #dummy value for past electricity prices
        past_prices = np.random.uniform(low=0.05, high=0.20, size=24)  #random prices for the past 24 hours within a given range
        return np.array([ut, Et] + list(past_prices)) #return the state

    def step(self, action):
        ut, Et = self.state[0], self.state[1] #get current state for home status and battery level
        new_Et = np.clip(Et + action, 0, self.battery_capacity) #perform action, get new battery level, np.clip restricts the range
        #action can be positive or negative, positive is charhing, negative is selling (recharhing)
        cost = self.electricity_prices[-1] * action #apply reward
        
        reward = -cost  #goal is minimizing the payings for the electricity
        
        #Update the electricity prices to simulate time passing
        self.electricity_prices = np.roll(self.electricity_prices, -1) #shift the prices array
        self.electricity_prices[-1] = self.get_new_price()  #assume a function to get the new price
        
        #Update the state by applying new battery level and electricity prices
        self.state = np.array([ut, new_Et] + list(self.electricity_prices))
        
        return self.state, reward
    
    def get_new_price(self):
        #Placeholder for obtaining new electricity price
        #In practice, this would come from a predictive model or the environment
        return np.random.uniform(low=0.05, high=0.20)

env = V2GEnvironment(max_charge_rate=10, max_discharge_rate=-10, battery_capacity=100, electricity_prices=np.random.uniform(0.05, 0.20, size=24))
state = env.reset()
for step in range(10):  # Simulate 10 steps
    action = np.random.uniform(env.max_discharge_rate, env.max_charge_rate)  # Random action
    state, reward = env.step(action)
    
    #Determine if the action is a charge or discharge
    action_type = 'Charge' if action > 0 else 'Discharge'
    
    print(f"Step {step + 1}:")
    print(f"Action taken: {action_type} ({action:.2f} units)")
    print(f"Reward gained: {reward:.2f}")
    print(f"New battery level: {state[1]:.2f} units")
    print(f"Electricity price for the last hour: {env.electricity_prices[-1]:.2f}")
    print(f"Total state vector: {state}")
    print("---")
