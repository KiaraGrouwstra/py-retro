from agents.agent import Agent
import tensorflow as tf
from summ import summary_scalar, summary_tensor, summary_histogram, summary_value

class RandomTFAgent(Agent):
    def __init__(self, client, instance_id, **kwargs):
        super(RandomTFAgent, self).__init__(client, instance_id)
        self.sess = tf.Session()
        self.sess.as_default()
        self.dist = tf.distributions.Bernoulli(probs=[0.5], name='Bernoulli')
        self.writer = tf.summary.FileWriter('/tmp/log/', self.sess.graph)
        self.action = self.dist.sample(self.shape)
        summary_histogram(self.action)
        # tf.Summary.Value(simple_value=self.action, tag='Action_Summary')
        # self.t = tf.placeholder(tf.int32, shape=[], name='t')
        self.reward = tf.placeholder(tf.int32, shape=[], name='reward')
        # self.cum_reward = tf.placeholder(tf.int32, shape=[], name='cum_reward')
        self.summary_op = tf.summary.merge_all()
        summary = self.sess.run(self.summary_op) # , feed_dict={self.reward:0} # , self.t:0
        global_step = tf.train.get_global_step()
        self.writer.add_summary(summary, global_step)

        self.tot_reward = 0

    def __del__(self):
        self.writer.close()
        self.sess.close()

    def act(self, obs, **kwargs):
        return self.action.eval(session=self.sess)
    
    def learn(self, state, action, reward, next_state, done, **kwargs):
        self.tot_reward += reward
        _ = self.sess.run([], feed_dict={self.reward:reward}) # , self.t:kwargs['t']
        episode_summary = tf.Summary(value=[
            tf.Summary.Value(tag='Reward', simple_value=reward),
        ])
        global_step = tf.train.get_global_step()
        self.writer.add_summary(episode_summary, global_step)
