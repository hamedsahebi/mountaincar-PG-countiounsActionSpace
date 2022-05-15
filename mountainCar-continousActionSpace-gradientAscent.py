import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

tf.compat.v1.disable_eager_execution()


class FeatureTransformer:

    def __init__(self, env,n_components=500):

        observation_examples = np.array(
            [env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
        ])

        feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimensions = feature_examples.shape[1]
        

        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):

        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)



class HiddenLayer:
  def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True, zeros=False):
    if zeros:
      W = np.zeros((M1, M2), dtype=np.float32)
    else:
      W = tf.random.normal(shape=(M1, M2)) * np.sqrt(2. / M1, dtype=np.float32)
    self.W = tf.Variable(W)

    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(M2).astype(np.float32))

    self.f = f

  def forward(self, X):
    if self.use_bias:
      a = tf.matmul(X, self.W) + self.b
    else:
      a = tf.matmul(X, self.W)
    return self.f(a)


# approximates pi(a | s)
class PolicyModel:

    def __init__(self, D, ft, hidden_layer_sizes=[]):
        self.ft = ft

        ##### hidden layers #####
        M1 = D
        self.hidden_layers = []
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.hidden_layers.append(layer)
            M1 = M2

        # final layer mean
        self.mean_layer = HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)

        # final layer variance
        self.stdv_layer = HiddenLayer(M1, 1, tf.nn.softplus, use_bias=False, zeros=False)

        # inputs and targets
        self.X = tf.compat.v1.placeholder(tf.float32, shape=(None, D), name='X')
        self.actions = tf.compat.v1.placeholder(tf.float32, shape=(None,), name='actions')
        self.advantages = tf.compat.v1.placeholder(tf.float32, shape=(None,), name='advantages')

        # get final hidden layer
        Z = self.X
        for layer in self.hidden_layers:
            Z = layer.forward(Z)

        # calculate output and cost
        mean = self.mean_layer.forward(Z)
        stdv = self.stdv_layer.forward(Z) + 1e-5 # smoothing

        # make them 1-D
        mean = tf.reshape(mean, [-1])
        stdv = tf.reshape(stdv, [-1]) 

        norm = tf.compat.v1.distributions.Normal(mean, stdv)
        self.predict_op = tf.clip_by_value(norm.sample(), -1, 1)

        log_probs = norm.log_prob(self.actions)
        cost = -tf.reduce_sum(input_tensor=self.advantages * log_probs + 0.1*norm.entropy())
        self.train_op = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(cost)
    
    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, actions, advantages):

        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        self.session.run(
        self.train_op,
        feed_dict={
            self.X: X,
            self.actions: actions,
            self.advantages: advantages,
        }
        )

    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def sample_action(self, X):
        p = self.predict(X)[0]
        return p

# approximates V(s)
class ValueModel:

    def __init__(self, D, ft, hidden_layer_sizes=[]):

        self.ft = ft
        self.costs = []

        # create the graph
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # final layer
        layer = HiddenLayer(M1, 1, lambda x: x)
        self.layers.append(layer)

        # inputs and targets
        self.X = tf.compat.v1.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.compat.v1.placeholder(tf.float32, shape=(None,), name='Y')

        # calculate output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
            Y_hat = tf.reshape(Z, [-1]) # the output
        self.predict_op = Y_hat

        cost = tf.reduce_sum(input_tensor=tf.square(self.Y - Y_hat))
        self.cost = cost
        self.train_op = tf.compat.v1.train.AdamOptimizer(1e-1).minimize(cost)

    def set_session(self,session):
        self.session = session

    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        Y = np.atleast_1d(Y)
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})
        cost = self.session.run(self.cost, feed_dict={self.X: X, self.Y: Y})
        self.costs.append(cost)

    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

def play_one_td(env, pmodel, vmodel, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 2000:
        # if we reach 2000, just quit, don't want this going forever
        # the 200 limit seems a bit early
        action = pmodel.sample_action(observation)
        prev_observation = observation
        observation, reward, done, info = env.step([action])

        totalreward += reward

        # update the models
        V_next = vmodel.predict(observation)
        G = reward + gamma*V_next
        advantage = G - vmodel.predict(prev_observation)
        pmodel.partial_fit(prev_observation, action, advantage)
        vmodel.partial_fit(prev_observation, G)

        iters += 1

    return totalreward, iters

def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(
        env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(
        env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)

    Z = np.apply_along_axis(
        lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)

    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title("Cost-to-go Function")
    fig.colorbar(surf)
    plt.show()


def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)

    for t in range(N):
        running_avg[t] = total_rewards[max(0, t-100):t+1].mean()

    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

def main():
    env = gym.make('MountainCarContinuous-v0')
    ft = FeatureTransformer(env, n_components=100)
    D = ft.dimensions
    pmodel = PolicyModel(D, ft, [])
    vmodel = ValueModel(D, ft, [])
    init = tf.compat.v1.global_variables_initializer()
    session = tf.compat.v1.InteractiveSession()
    session.run(init)
    pmodel.set_session(session)
    vmodel.set_session(session)
    gamma = 0.95

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 100
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        totalreward, num_steps = play_one_td(env, pmodel, vmodel, gamma)
        totalrewards[n] = totalreward
        if n % 10 == 0:
            print("episode:", n, "total reward: %.1f" % totalreward, "num steps: %d" % num_steps, "avg reward (last 100): %.1f" % totalrewards[max(0, n-100):(n+1)].mean())

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)
    plot_cost_to_go(env, vmodel)

if __name__ == "__main__":
    main()


        


