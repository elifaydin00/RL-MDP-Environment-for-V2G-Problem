import numpy as np
import datetime

class V2GEnvironment:
    def __init__(self, max_charge_rate, max_discharge_rate, battery_capacity, electricity_prices, gamma=0.99):
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.battery_capacity = battery_capacity
        self.electricity_prices = electricity_prices
        self.rewards = 0
        self.gamma = gamma
        self.state = self.reset() #initial configuration state
    
    #state contains whether ev is at home or not, current battery level and past 24 hr prices and a time variable
    def reset(self):
        # Initialize the state based on the problem description
        ut = 1 #ut is set to 1 to indicate the EV is at home
        Et = self.battery_capacity / 2 #battery energy starts at half capacity
        past_prices = np.random.uniform(low=0.05, high=0.20, size=24)  #random prices for the past 24 hours within a given range
        time_variable = datetime.datetime(2024, 1, 1, 0, 0)  #Set the time variable to January 1, 2024 at 00:00
        hour = time_variable.hour  #Extract the hour (though it will be 0 in this case)
        return {"status": [ut, Et], "time": time_variable, "prices": past_prices} #State

    def step(self, action):
        ut, Et = self.state["status"]  # Get current status for home and battery level

        #Morning commute minimum %80 Constraint
        next_time = self.state["time"] + datetime.timedelta(hours=1)  #Get next time

        if next_time.hour == 8 and Et < 0.8 * self.battery_capacity:
            needed_charge = 0.8 * self.battery_capacity - Et
            if(needed_charge > action):
                action = needed_charge #Ensure enough charging action to meet the requirement

        # Always keep a minimum of 20% battery level Constraint
        min_required = 0.2 * self.battery_capacity
        if Et + action < min_required:
            action = min_required - Et  # Adjust action to maintain minimum battery level


        new_Et = np.clip(Et + action, 0, self.battery_capacity)  # Perform action and get new battery level

        cost = self.electricity_prices[-1] * action  # Gained Reward calculation = action * last price
        self.rewards -= cost  # Update cumulative rewards

        # Update the electricity prices to simulate time passing
        self.electricity_prices = np.roll(self.electricity_prices, -1)
        self.electricity_prices[-1] = self.get_new_price()
        
        #Increment the hour by 1
        new_time = self.state["time"] + datetime.timedelta(hours=1)

        #Update the state with new battery level, time, and electricity prices
        self.state = {"status": [ut, new_Et], "time": new_time, "prices": self.electricity_prices}
        
        return self.state, self.rewards, action
    
    def get_new_price(self):
        #Placeholder for obtaining new electricity price
        #In practice, this would come from a predictive model or the environment
        return np.random.uniform(low=0.05, high=0.20)

env = V2GEnvironment(max_charge_rate=10, max_discharge_rate=-10, battery_capacity=100, electricity_prices=np.random.uniform(0.05, 0.20, size=24))
state = env.reset()
for step in range(10):  # Simulate 10 steps
    action = np.random.uniform(env.max_discharge_rate, env.max_charge_rate)  # Random action
    state, reward, act = env.step(action)

    # Determine if the action is a charge or discharge
    action_type = 'Charge' if act > 0 else 'Discharge'
    
    # Extract data from the state dictionary
    ut, new_Et = state["status"]  # Updated vehicle status at home and battery level

    print(f"Step {step + 1}:")
    print(f"Action taken: {action_type} ({act:.2f} units)")
    print(f"Total reward: {reward:.2f}")
    print(f"New battery level: {new_Et:.2f} units")
    print(f"Current Date-Time: {state['time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total state vector: {state}")
    print("---")
