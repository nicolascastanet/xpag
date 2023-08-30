import numpy as np


def sample_random_buffer(buffer, batch_size):
    buffers = buffer.pre_sample()
    rollout_batch_size = buffers["episode_length"].shape[0] # buffer current size
    
    episode_idxs = np.random.choice(
        rollout_batch_size,
        size=batch_size,
        replace=True,
        p=buffers["episode_length"][:, 0, 0]
        / buffers["episode_length"][:, 0, 0].sum(),
                )

    t_max_episodes = buffers["episode_length"][episode_idxs, 0].flatten()
    t_samples = np.random.randint(t_max_episodes)
    transitions = {
            key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()
        }
    
    return transitions


def softmax(X, theta=1.0, axis=None):
  """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

  # make X at least 2d
  y = np.atleast_2d(X)

  # find axis
  if axis is None:
    axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

  # multiply y against the theta parameter,
  y = y * float(theta)

  # subtract the max for numerical stability
  y = y - np.max(y, axis=axis, keepdims=True)

  # exponentiate y
  y = np.exp(y)

  # take the sum along the specified axis
  ax_sum = np.sum(y, axis=axis, keepdims=True)

  # finally: divide elementwise
  p = y / ax_sum

  # flatten if X was 1D
  if len(X.shape) == 1: p = p.flatten()

  return p