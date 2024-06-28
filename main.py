from env import Auv_SimpleAction, Auv_MultiActions,Auv_changeable,Auv_blined
from policy import DQNAgent, preprocess_state
import torch


class AUV_config:
    def __init__(self, env_mode, batch_size, buffer_capacity, max_fuels):
        self.env_mode = env_mode # simple / Complex
        
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.lr = 0.001
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.max_fuels = max_fuels 

        self.episodes = 10000
        self.target_update_frequency = 1000
        self.visual_frequency = 4000

        self.use_prioritise_buffer = False
        self.save_freq = 5000

        self.model_save_path = 'result/' + str(self.env_mode) + '-' + str(self.buffer_capacity) + '-' + str(self.target_update_frequency) +'--model'+ '.pth'
        self.target_model_save_path = 'result/' + str(self.env_mode) + '-' + str(self.buffer_capacity) + '-' + str(self.target_update_frequency) +'--target_model'+ '.pth'
        self.image_save_path = 'assets/' +str(self.env_mode)
        
config = AUV_config('Simple', batch_size=64, buffer_capacity=1000, max_fuels=500)
# config = AUV_config('Complex', batch_size=64, buffer_capacity=10000, max_fuels=200)

# env = Auv_SimpleAction(config.max_fuels, config.image_save_path) if config.env_mode == 'Simple' else Auv_MultiActions(config.max_fuels, config.image_save_path)
# env = Auv_changeable(config.max_fuels, config.image_save_path)
env = Auv_blined(config.max_fuels, config.image_save_path)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
print(state_dim, action_dim)
#------------------------------------train--------------------------
agent = DQNAgent(config.use_prioritise_buffer, state_dim,action_dim,config.buffer_capacity, config.batch_size, gamma= 1, lr = config.lr)

steps = 0

for episode in range(config.episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        steps += 1
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        
        agent.remember(state, action, reward, next_state, done)
        loss = agent.replay()
        state = next_state
        total_reward += reward

        if steps % config.target_update_frequency == 0:
            agent.update_target_network()

    if episode >= 1 and agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
    
    # if episode and episode % config.visual_frequency == 0:
    #     env.render()
    #     env.visualize_metrics()
    
    if episode and episode % config.save_freq == 0:
        env.render()
        env.visualize_metrics()
        agent.save_model(config.model_save_path, config.target_model_save_path)
    
    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}, Loss: {loss}")

agent.save_model(config.model_save_path, config.target_model_save_path)
env.render()
env.visualize_metrics()


# -------------------------------------test----------------------------------------
# agent_loaded = DQNAgent(config.use_prioritise_buffer, state_dim,action_dim,config.buffer_capacity, config.batch_size, gamma= 1, lr = config.lr)


# # Load the saved model parameters
# agent_loaded.model.load_state_dict(torch.load("result\q2.pth"))
# agent_loaded.model.eval()  # Set the model to evaluation mode
# state = env.reset()
# done = False
# agent_loaded.epsilon = 0
# while not done:
#     action = agent_loaded.act(state)
#     next_state, reward, done = env.step(action)
#     state = next_state

# env.render_gif()
