import tensorflow as tf
import fenchel_young
import perturbations


def ranks_fn(x, axis=-1):
    return tf.cast(tf.argsort(tf.argsort(x, axis=axis), axis=axis), dtype=x.dtype)



def RankingLoss(num_samples=1000000, sigma=0.1, noise='gumbel', batched=True):
    return fenchel_young.FenchelYoungLoss(ranks_fn, num_samples, sigma, noise, batched)

class Ranking(tf.keras.layers.Layer):
    def __init__(self, num_samples=1000000, sigma=0.1, noise='gumbel', batched=True):
        self.num_samples = num_samples
        self.sigma = sigma
        self.noise = noise
        self.batched=batched
        super(Ranking, self).__init__()
    def call(self, inputs):
        @perturbations.perturbed(num_samples=self.num_samples, sigma=self.sigma, noise=self.noise, batched=self.batched)
        def ranks_fn(x, axis=-1):
            return tf.cast(tf.argsort(tf.argsort(x, axis=axis), axis=axis), dtype=x.dtype)
        return ranks_fn(inputs)

class PartialRanksAccuracy(tf.keras.metrics.Mean):
  """This metric the proportion of matching ranks."""

  def __init__(self, name='partial_ranks_acc'):
    super().__init__(name=name)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.cast(ranks_fn(y_pred, axis=-1), tf.int32)
    y_true = tf.cast(y_true, tf.int32)
    equals = tf.cast(y_true == y_pred, tf.float32)
    result = tf.math.reduce_mean(equals, axis=-1)
    super().update_state(
        tf.reduce_mean(result, axis=-1), sample_weight=sample_weight)

class ProjectedRanksAccuracy(tf.keras.metrics.Mean):
  """This metric is the normalized projection onto the permutahedron."""

  def __init__(self):
    super().__init__(name='projection_ranks_acc')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.cast(ranks_fn(y_pred, axis=-1), tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    n = tf.cast(tf.shape(y_true)[-1], tf.float32)
    max_proj = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0
    result = tf.math.reduce_sum(y_true * y_pred, axis=-1) / max_proj
    super().update_state(
        tf.reduce_mean(result, axis=-1), sample_weight=sample_weight)



def main():
    myloss = RankingLoss()
    a = tf.convert_to_tensor([[1.,2,0],[0.,1,2]])
    b = tf.convert_to_tensor([[0.5,1,0.2],[0.,1,1.2]])
    print('Loss of', a, b, 'is', myloss(a,b).numpy())
    a = tf.convert_to_tensor([[1.,2,0],[0.,1,2]])
    b = tf.convert_to_tensor([[1,2,0.2],[0.,1,1.2]])
    print('Loss of', a, b, 'is', myloss(a,b).numpy())
    a = tf.convert_to_tensor([[1.,2,0],[0.,1,2]])
    b = tf.convert_to_tensor([[10,20,00],[00.,10,20]])
    print('Loss of', a, b, 'is', myloss(a,b).numpy())
    print(tf.convert_to_tensor([[1.,0.2,3],[0.,1,2]]), 'as ranks:', ranks_fn(tf.convert_to_tensor([[1.,0.2,3],[0.,1,2]])))
    return None

if __name__=='__main__':
    main()