import tensorflow as tf
from source import layers

'''
i want to use this network.py as a parser to load
.json files containing network parameters
'''
def _get_options(spec):
    # some try/except blocks
    try:
        padding = spec['padding'].encode('utf-8')
    except KeyError:
        padding = 'SAME'

    # eval() only for now
    try:
        kernel_initializer = eval(spec['kernel_initializer'])
    except KeyError:
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.01)

    try:
        bias_initializer = eval(spec['bias_initializer'])
    except KeyError:
        bias_initializer = tf.constant_initializer(value=0.01)

    # this is a minor change, because I want to include
    # activation = None in .json for clarity sometimes
    try:
        activation = eval(spec['activation'])
    except KeyError:
        activation = None

    opts = {'padding': padding, 
            'kernel_initializer': kernel_initializer, 
            'bias_initializer': bias_initializer, 
             'activation': activation}
    return opts

def _build_conv2d(inputs, spec, name):
    opts = _get_options(spec)

    inputs = layers.conv2d(inputs=inputs,
                           filters=spec['filters'],
                           kernel_size=spec['kernel_size'],
                           stride_size=spec['stride_size'],
                           padding=opts['padding'],
                           kernel_initializer=opts['kernel_initializer'],
                           bias_initializer=opts['bias_initializer'],
                           activation=opts['activation'],
                           name=name)
    return inputs

def _build_dense(inputs, spec, name):
    opts = _get_options(spec)

    inputs = layers.dense(inputs=inputs,
                          units=spec['units'],
                          kernel_initializer=opts['kernel_initializer'],
                          bias_initializer=opts['bias_initializer'],
                          activation=opts['activation'],
                          name=name)
    return inputs

def build_network(inputs, 
                  model_specs, 
                  latent_dim=None,
                  num_channel=None, 
                  name='network'):
    # automatically handle names
    num = {}

    with tf.variable_scope(name):
        for block in model_specs:
            for name, spec in block.iteritems():
                if name == 'conv2d':
                    if not name in num.keys():
                        num[name] = 1

                    # for automatic build of output channels
                    if spec['filters'] == 'num_channel':
                        spec['filters'] = num_channel
                        
                    inputs = _build_conv2d(inputs=inputs, 
                                           spec=spec,
                                           name='{}_{}'.format(name, num[name]))
                    num[name] += 1

                elif name == 'flatten' and spec:
                    inputs = layers.flatten(inputs=inputs)

                elif name == 'spatial_broadcast':
                    inputs = layers.spatial_broadcast(z=inputs,
                                                      w=spec['w'],
                                                      h=spec['h'],
                                                      name='spatial_broadcast')

                elif name == 'dense':
                    if not name in num.keys():
                        num[name] = 1
                    if spec['units'] == 'latent_dim':
                        spec['units'] = latent_dim
                    inputs = _build_dense(inputs=inputs,
                                          spec=spec,
                                          name='{}_{}'.format(name, num[name]))
                    num[name] += 1
                else:
                    raise TypeError('undefined name for a network element: {}'.format(name))
        return inputs