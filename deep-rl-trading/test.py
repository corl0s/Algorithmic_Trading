# from src.rl.agent import Agent


# agent = Agent("transforming-stonks-v1.0.1", tickers=["GOOGL"])
# agent.test()




# test.py

# from src.rl.agent import Agent
# import numpy as np

# # 1. Define the model name
# MODEL_NAME = "transforming-stonks-v1.0.1" 

# # 2. Initialize the agent (which sets up the environment and network architecture)
# agent = Agent(MODEL_NAME, tickers=["GOOGL"])

# # 3. **CRITICAL STEP:** Load the trained weights from the saved file.
# #    It assumes your model is saved at 'models/transforming-stonks-v1.0.1'
# try:
#     agent.load(MODEL_NAME)
#     print(f"Successfully loaded model: {MODEL_NAME}")
# except FileNotFoundError:
#     print(f"Error: Model file 'models/{MODEL_NAME}' not found.")
#     # Exit or handle the error if the model isn't there
#     exit()

# # 4. Run the test method
# rewards = agent.test()

# # 5. Print/log the results
# print("\n--- Testing Complete ---")
# print(f"Total rewards from test stocks: {rewards}")
# print(f"Average reward: {np.mean(rewards)}") # You'll need to import numpy in test.py for this





from src.rl.agent import Agent
import numpy as np # Keep this for calculating the mean

MODEL_NAME = "transforming-stonks-v1.0.1" 

agent = Agent(MODEL_NAME, tickers=["GOOGL"]) # The tickers here are just for initialization

try:
    agent.load(MODEL_NAME)
    print(f"Successfully loaded model: {MODEL_NAME}")
except FileNotFoundError:
    print(f"Error: Model file 'models/{MODEL_NAME}' not found.")
    exit()

rewards = agent.test()

print("\n--- Testing Complete ---")
print(f"Total rewards from test stocks: {rewards}")
print(f"Average reward: {np.mean(rewards)}")