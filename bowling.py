import sys
import gym
import pylab
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from collections import deque
from keras.layers import Dense

num_episodes = 50

class Agent:
    def __init__(self, state_len, action_len):
        self.render = False
        self.model_trained = True

        # 상태와 행동의 크기 정의
        self.state_len = state_len
        self.action_len = action_len

        # DQN 하이퍼파라미터
        self.disc_factor = 0.99
        self.lr = 0.001
        self.eps = 1.0
        self.eps_decay = 0.9995
        self.eps_min = 0.01
        self.batch_size = 64
        self.train_min = 1000

        # 리플레이 메모리, 최대 크기 2000
        self.buffer = deque(maxlen=5000)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.targ_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()

        if self.model_trained:
            self.model.load_weights("./save_model/bowling_trained.h5py")

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_len, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(12, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_len, activation='linear',
                        kernel_initializer='he_uniform'))
        #model.summary()
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))
        return model

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.targ_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.eps:
            return random.randrange(self.action_len)
        else:
            q_val = self.model.predict(state)
            return np.argmax(q_val[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        m_batch = random.sample(self.buffer, self.batch_size)

        state_lst = np.zeros((self.batch_size, self.state_len))
        next_state_lst = np.zeros((self.batch_size, self.state_len))
        action_lst, reward_lst, done_lst = [], [], []

        for i in range(self.batch_size):
            state_lst[i] = m_batch[i][0]
            action_lst.append(m_batch[i][1])
            reward_lst.append(m_batch[i][2])
            next_state_lst[i] = m_batch[i][3]
            done_lst.append(m_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        tar = self.model.predict(state_lst)
        tar_val = self.targ_model.predict(next_state_lst)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if done_lst[i]:
                tar[i][action_lst[i]] = reward_lst[i]
            else:
                tar[i][action_lst[i]] = reward_lst[i] + self.disc_factor * (
                    np.amax(tar_val[i]))

        self.model.fit(state_lst, tar, batch_size=self.batch_size,
                       epochs=1, verbose=0)

if __name__ == "__main__":
    env = gym.make("ALE/Bowling-v5")#, render_mode='human')
    state_len = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    action_len = env.action_space.n

    # DQN 에이전트 생성
    agent = Agent(state_len, action_len)

    score_lst, episode_lst = [], []

    for e in range(num_episodes):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state[0], [1, state_len])

        while not done:
            if agent.render:
                env.render()

            # 현재 상태로 행동을 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_len])
            # 에피소드가 중간에 끝나면 -100 보상
            #reward = reward if not done or score == 499 else -100

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            agent.append_sample(state, action, reward, next_state, done)
            # 매 타임스텝마다 학습
            if len(agent.buffer) >= agent.train_min:
                agent.train_model()

            score += reward
            state = next_state

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()

                #score = score if score == 500 else score + 100
                # 에피소드마다 학습 결과 출력
                score_lst.append(score)
                episode_lst.append(e)
                pylab.plot(episode_lst, score_lst, 'b')
                pylab.savefig("./save_graph/bowling_graph.png")
                print("episode:", e, "  score:", score, "  buffer length:",
                      len(agent.buffer), "  epsilon:", agent.eps)

                # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
                if score > 29:
                    agent.model.save_weights("./save_model/bowling_trained.h5py")
                    sys.exit()


    print("Number of successful episodes: %d / %d"%(reward, num_episodes))
    print("Average number of timesteps per episode: %.2f"%(episode_lst/num_episodes))
    