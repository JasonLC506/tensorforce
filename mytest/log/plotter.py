from matplotlib import pyplot as plt
import numpy as np

file_name = "./log"
filter = True
N = 100

logs = []
log_episode = []
with open(file_name, "r") as f:
    for line in f:
        line = line.rstrip("\n")
        if "episode" in line:
            if len(log_episode) == 3:
                logs.append(log_episode)
            log_episode = []
            episode = line.split(" ")[-1]
            log_episode.append(int(episode))
        elif "reward" in line:
            reward = line.split(" ")[-1]
            log_episode.append(float(reward))
            assert len(log_episode) == 2
        elif "timesteps" in line:
            timesteps = line.split(" ")[-1]
            log_episode.append(int(timesteps))
            assert len(log_episode) == 3
        else:
            pass
logs = np.array(logs)
if filter:
    logs_new = [logs[:, 0]]
    for i in range(1, logs.shape[-1]):
        logs_new.append(np.convolve(logs[:, i], np.ones(N)/N, 'same')[:logs_new[0].shape[0]])
    logs = np.array(logs_new).transpose()
plt.plot(logs[:, 0], logs[:, 1], label="reward", linewidth=0.2)
plt.plot(logs[:, 0], logs[:, 2], label="timesteps", linewidth=0.1)
plt.legend()
plt.show()