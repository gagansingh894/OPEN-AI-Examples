import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from statistics import mean, median
from collections import Counter
import tensorflow as tf

tf.config.experimental.set_visible_devices([], 'GPU')

# LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

#create random games

def some_random_games_first():
	for episode in range(50):
		env.reset()
		for t in range(goal_steps):
			env.render()
			action = env.action_space.sample() # take a random action in the environment
			observation, reward, done, info = env.step(action) # observation - array of data from game eg: pixel data | reward - 1 or 0 | done = state 1 or 0 | # info some info
			if done:
				break

# some_random_games_first()

def initial_population():
	
	training_data = [] # THE MOVES MADE	
	scores = []
	accepted_scores = []
	for _ in range(initial_games):
		score = 0
		game_memory = []
		prev_observation = []
		for _ in range(goal_steps):
			# env.render()
			action = random.randrange(0,2)
			observation, reward, done, info = env.step(action)

			if len(prev_observation) > 0:
				game_memory.append([prev_observation, action])

			prev_observation = observation
			score += reward
			if done:
				break

		#Analyze the game

		if score >= score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				if data[1] == 1:
					output = [0,1]
				elif data[1] == 0:
					output = [1,0]

				training_data.append([data[0], output])

		env.reset()
		scores.append(score)

	
	training_data_save = np.array(training_data)
	print(training_data_save.shape)
	np.save('saved.npy', training_data_save)

	print('Average accepted score:', mean(accepted_scores))
	print('Median accepted score:', median(accepted_scores))
	print(Counter(accepted_scores))
	print(len(accepted_scores))
	return training_data_save

def neuaral_network_model(input_dim):
	model = Sequential()
	model.add(Dense(units=128, input_shape=(input_dim,), activation='relu'))
	model.add(Dropout(rate=0.2))

	model.add(Dense(units=256, activation='relu'))
	model.add(Dropout(rate=0.2))

	model.add(Dense(units=512, activation='relu'))
	model.add(Dropout(rate=0.2))

	model.add(Dense(units=256, activation='relu'))
	model.add(Dropout(rate=0.2))

	model.add(Dense(units=128, activation='relu'))
	model.add(Dropout(rate=0.2))

	model.add(Dense(units=2, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model


def train_model(data):
	X = np.stack(data[:,0])
	y = np.stack(data[:,1])
	model = neuaral_network_model(4) #need to make this dynamic for future
	model.fit(X, y, epochs=1, batch_size=32)
	return model


training_data = initial_population()
model = train_model(training_data)
# print(type(training_data))
# print(y[0:5])


scores = []
choices = []

for each_game in range(10):
	score = 0
	game_memory = []
	prev_obs = []
	env.reset()
	for _ in range(goal_steps):
		env.render()
		if len(prev_obs) == 0:
			action = random.randrange(0,2)
		else:
			print(np.array(prev_obs))
			action = np.argmax(model.predict(np.array(prev_obs).reshape(-1,4))[0])
			print(action)
		choices.append(action)
		
		new_observation, reward, done, info = env.step(action)
		prev_obs = new_observation
		game_memory.append([new_observation,action])
		score += reward
		if done:
			break
	scores.append(score)

print('Average Score', sum(scores)/len(scores))
print('Choice 1: {}, Choice 0: {}'. format(choices.count(1)/len(choices),
	choices.count(0)/len(choices)))
