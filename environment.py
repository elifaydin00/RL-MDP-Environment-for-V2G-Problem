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
        time_variable = datetime.datetime(2024, 1, 1, 0, 0)  #Set the time variable to January 1, 2024 at 00:00
        hour = time_variable.hour  #Extract the hour (though it will be 0 in this case)
        return {"status": [ut, Et], "time": time_variable, "prices": self.electricity_prices} #State

    def printStep(self, step, action, reward, new_Et, state):
        action_type = 'Charge' if action > 0 else 'Discharge'
        
        # Extract data from the state dictionary
        ut, new_Et = state["status"]  #Vehicle status at home and battery level

        print(f"Step {step + 1}:")
        print(f"Action taken: {action_type} ({action:.2f} units)")
        print(f"Total reward: {reward:.2f}")
        print(f"New battery level: {new_Et:.2f} units")
        print(f"Current Date-Time: {state['time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total state vector: {state}")
        print("---")

    def constraints(self, Et, next_time, action):

        #Morning and evening commute minimum %80 Constraint
        if next_time.hour == 8 or next_time.hour == 18 and Et < 0.8 * self.battery_capacity:
                needed_charge = 0.8 * self.battery_capacity - Et
                if(needed_charge > action):
                    action = needed_charge #Ensure enough charging action to meet the requirement

        # Always keep a minimum of 20% battery level Constraint
        #Improves battery lifecycle, emergency usage
        min_required = 0.2 * self.battery_capacity
        if Et + action < min_required:
            action = min_required - Et  # Adjust action to maintain minimum battery level

        return action



    def step(self):

        for i in range(24):  # Simulate 24 steps (hours)
            action = np.random.uniform(env.max_discharge_rate, env.max_charge_rate)  # Random action

            ut, Et = self.state["status"]  # Get current status for home and battery level

            #Get next time to apply constraints
            next_time = self.state["time"] + datetime.timedelta(hours=1)  #Get next time

            #Check constraints
            action = self.constraints(Et, next_time, action)
            new_Et = np.clip(Et + action, 0, self.battery_capacity)  # Perform action and get new battery level

            #Calculate the cost based on the price at the 0th index and the action taken
            cost = self.state["prices"][0] * action

            #Update cumulative rewards by subtracting the cost
            self.rewards -= cost

            #Shift the prices in the state to simulate time passing
            self.state["prices"] = np.roll(self.state["prices"], -1)

            #Update the last price in the shifted array with a new price from the method
            #self.state["prices"][-1] = self.get_new_price()
            
            #Increment the hour by 1
            new_time = self.state["time"] + datetime.timedelta(hours=1)

            #Update the state with new battery level, time, and electricity prices
            self.state = {"status": [ut, new_Et], "time": new_time, "prices": self.state["prices"]}

            self.printStep(i, action, self.rewards, new_Et, self.state)
        return self.rewards

    
    def get_new_price(self):
        #Placeholder for obtaining new electricity price
        #In practice, this would come from a predictive model or the environment
        return np.random.uniform(low=0.05, high=0.20)


past_prices = [
        0.06, 0.06, 0.06, 0.06, 0.06,  # 00:00 to 04:00 - Low (night time)
        0.07, 0.07,  # 05:00 to 06:00 - Increasing (early risers)
        0.09, 0.11, 0.11, 0.10, 0.10,  # 07:00 to 11:00 - High (morning commute)
        0.08, 0.08, 0.08, 0.08,  # 12:00 to 15:00 - Moderate (daytime)
        0.09, 0.09,  # 16:00 to 17:00 - Increasing (late afternoon)
        0.12, 0.15, 0.18, 0.18,  # 18:00 to 21:00 - Peak (evening commute and usage)
        0.12, 0.10, 0.08  # 22:00 to 23:00 - Decreasing (night begins)
        ]


env = V2GEnvironment(max_charge_rate=10, max_discharge_rate=-10, battery_capacity=100, electricity_prices=past_prices)
number_of_runs = 2
day_rewards = [0 for _ in range(number_of_runs)]
for run in range(number_of_runs):
    #TO FIX Problem on reset function
    #state = env.reset() 
    env = V2GEnvironment(max_charge_rate=10, max_discharge_rate=-10, battery_capacity=100, electricity_prices=past_prices)
    run_reward= env.step()
    day_rewards[run] = run_reward

print("Results of day rewards: ",day_rewards)

