import tensorflow as tf

def summary_scalar(x):
    tf.summary.scalar(x.op.name + '_summary', x)

def summary_tensor(x):
    tf.summary.tensor_summary(x.op.name + '_summary', x)

def summary_histogram(x):
    tf.summary.histogram(x.op.name + '_summary', x)

def summary_value(x):
    tf.Summary.Value(tag=x.op.name + '_summary', simple_value=x)
