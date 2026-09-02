import math
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

cluster_size = 32
feature_size = 512
ghost_clusters = 0

class Model(keras.Model):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(Model, self).__init__()
        
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters
        
        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters
        
        self.clusters = tf.Variable(init_sc * tf.random.normal((feature_size, clusters)))
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.clusters2 = tf.Variable(init_sc * tf.random.normal((1, feature_size, cluster_size)))
        self.out_dim = self.cluster_size * feature_size
    
    def call(self, x, mask=None):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (tf.Tensor): B x N x D

        Returns:
            (tf.Tensor): B x DK
        """
        max_sample = tf.shape(x)[1]
        x = tf.reshape(x, [-1, self.feature_size])  # B x N x D -> BN x D

        if x.device != self.clusters.device:
            raise ValueError(f"x.device {x.device} != cluster.device {self.clusters.device}")

        assignment = tf.matmul(x, self.clusters)  # (BN x D) x (D x (K+G)) -> BN x (K+G)
        assignment = self.batch_norm(assignment)

        assignment = tf.nn.softmax(assignment, axis=1)  # BN x (K+G) -> BN x (K+G)
        # remove ghost assigments
        assignment = assignment[:, :self.cluster_size]
        assignment = tf.reshape(assignment, [-1, max_sample, self.cluster_size])  # -> B x N x K
        a_sum = tf.reduce_sum(assignment, axis=1, keepdims=True)  # B x N x K -> B x 1 x K
        a = a_sum * self.clusters2

        assignment = tf.transpose(assignment, perm=[0, 2, 1])  # B x N x K -> B x K x N

        x = tf.reshape(x, [-1, max_sample, self.feature_size])  # BN x D -> B x N x D
        vlad = tf.matmul(assignment, x)  # (B x K x N) x (B x N x D) -> B x K x D
        vlad = vlad - a

        # L2 intra norm
        vlad = tf.math.l2_normalize(vlad)

        # flattening + L2 norm
        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.feature_size])  # -> B x DK
        vlad = tf.math.l2_normalize(vlad)
        return vlad  # B x DK

batch_size = 2048
num_features = 100
feature_size = 512

def get_inputs():
  return [tf.random.normal([batch_size, num_features, feature_size])]

def get_init_inputs():
  return [num_clusters, feature_size, ghost_clusters]
