import tensorflow as tf
def get_Intensity(U):
    I = tf.multiply(tf.math.conj(U), U)
    I = tf.math.real(I)
    maxnum = tf.reduce_max(I)
    minnum = tf.reduce_min(I)
    # M = tf.divide(I,maxnum)
    M = tf.divide((I - minnum), maxnum - minnum)
    M *= 255.
    return M