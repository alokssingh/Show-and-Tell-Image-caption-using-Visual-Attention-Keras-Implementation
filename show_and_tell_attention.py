# Example how to use this layer
#c_input = Input(shape=(seq_len,))
#v_input = Input(shape=(196,512))
#cells, h1, h2, att,  = Attentioncell(512,return_sequences = True, return_state=True, return_attention=True)([c_model_1,v_model_3],initial_state = [c_model,c_model])



from keras import backend as K
from keras.layers.recurrent import *
from keras.layers import TimeDistributed, Dense, LSTM, RNN
import tensorflow as tf



class Attentioncell(Layer):

    def __init__(self, units, return_sequences = False, return_state = False, go_backwards = False,stateful =False,
                 unroll = False,weight_initializer="glorot_uniform", return_attention=False,
                 dropout_W=0., dropout_U=0., name = 'cous', **kwargs):
        self.weight_initializer = weight_initializer
        self.return_attention = return_attention
        self.return_state = return_state
        self._num_constants = None        
        self.stateful = stateful
        self.units = units
        self.unroll =unroll
        self.name =name
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        #self.state_size = (self.units, self.units)
 
 
        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True

        super(Attentioncell, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]

    def _validate_input_shape(self, input_shape):
        if len(input_shape) >= 2:
            if len(input_shape[:2]) != 2:
                raise ValueError("Layer has to receive two inputs: the temporal signal and the external signal which is constant for all time steps")
            if len(input_shape[0]) != 3:
                raise ValueError("Layer received a temporal input with shape {0} but expected a Tensor of rank 3.".format(input_shape[0]))
            if len(input_shape[1]) != 3:
                raise ValueError("Layer received a time-independent input with shape {0} but expected a Tensor of rank 3.".format(input_shape[1]))
        else:
            raise ValueError("Layer has to receive at least 2 inputs: the temporal signal and the external signal which is constant for all time steps")



    def build(self, input_shape):
        self._validate_input_shape(input_shape)          
        #self.input_spec = [InputSpec(input_shape[0]), InputSpec(input_shape[1])]

        for i, x in enumerate(input_shape):
            self.input_spec[i] = InputSpec(shape=x)
        
        temporal_input_dim = input_shape[0][-1]
        static_input_dim = input_shape[1][-1]

        if self.return_sequences:
            output_dim = self.compute_output_shape(input_shape)[0][-1]
        else:
            output_dim = self.compute_output_shape(input_shape[0])[-1]

        self._W1 = self.add_weight(shape=(static_input_dim, temporal_input_dim), name="{}_W1".format(self.name), initializer=self.weight_initializer)
        self._W2 = self.add_weight(shape=(output_dim, temporal_input_dim), name="{}_W2".format(self.name), initializer=self.weight_initializer)
        self._W3 = self.add_weight(shape=(temporal_input_dim + static_input_dim, temporal_input_dim), name="{}_W3".format(self.name), initializer=self.weight_initializer)
        self._b2 = self.add_weight(shape=(temporal_input_dim,), name="{}_b2".format(self.name), initializer=self.weight_initializer)
        self._b3 = self.add_weight(shape=(temporal_input_dim,), name="{}_b3".format(self.name), initializer=self.weight_initializer)
        self._V = self.add_weight(shape=(temporal_input_dim, 1), name="{}_V".format(self.name), initializer=self.weight_initializer)
        super(Attentioncell, self).build(input_shape)

    @property
    def trainable_weights(self):
        return self._trainable_weights
    @property
    def non_trainable_weights(self):
        return self._non_trainable_weights 


    def compute_output_shape(self, input_shape):
        return super(Attentioncell, self).compute_output_shape(input_shape)

 
    def step(self, x, states):  
        #print(len(states))
        h = states[0]
        # states[1] necessary?
        # comes from the constants
        X_static = states[-2]
        # equals K.dot(static_x, self._W1) + self._b2 with X.shape=[bs, L, static_input_dim]
        total_x_static_prod = states[-1]

        # expand dims to add the vector which is only valid for this time step
        # to total_x_prod which is valid for all time steps
        hw = K.expand_dims(K.dot(h, self._W2), 1)
        additive_atn = total_x_static_prod + hw
        attention = K.softmax(K.dot(additive_atn, self._V), axis=1)
        static_x_weighted = K.sum(attention * X_static, [1])
        x = K.dot(K.concatenate([x, static_x_weighted], 1), self._W3) + self._b3
        h, new_states =  (x, states[:-2])
        attention = K.squeeze(attention, -1)
        h = K.concatenate([h, attention])
        return h, new_states
    
    def get_initial_states(self, x_input):
        mean_attn = Lambda(lambda x: K.mean(x, axis=1))
        mean_attn = mean_attn(x_input)
        h0 = K.dot(mean_attn, self.W_init_h) + self.b_init_h
        c0 = K.dot(mean_attn, self.W_init_c) + self.b_init_c
        return [self.attn_activation(h0), self.attn_activation(c0)]


    def compute_mask(self, input, mask):
        if self.return_sequences:
            return mask
        else:
            return None

    def call(self, x, constants=None, mask=None, initial_state=None):
        # input shape: (n_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape

        if len(x) > 2:
            initial_state = x[2:]
            x = x[:2]
            assert len(initial_state) >= 1

        static_x = x[1]
        x = x[0]

        if self.stateful:
            initial_states = self.states
        else:
            if initial_state is not None:
              initial_states = initial_state
              if not isinstance(initial_states, (list, tuple)):
                  initial_states = [initial_states]

        if mask is not None:
            mask = mask[0]
        else:
            mask = None

        if not constants:
            
            constants = []
        constants += self.get_constants(static_x)
        last_output, outputs, states = K.rnn(
            self.step,
            x,
            initial_states,
            go_backwards=self.go_backwards,
            mask=mask,
            constants=constants,
            unroll=self.unroll,
            input_length=input_shape[1]
        )
        output_dim = self.compute_output_shape(input_shape)[2]
        last_output = last_output[:output_dim]
        attentions = outputs[:, :, output_dim:]
        outputs = outputs[:, :, :output_dim]
        
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            output = outputs
        else:
            output = last_output 

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True

        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            output = [output] + states

        if self.return_attention:
            if not isinstance(output, list):
                output = [output]
            output = output + [attentions]

        return output

    def _standardize_args(self, inputs, initial_state, constants, num_constants):
        """Standardize `__call__` to a single list of tensor inputs.
        When running a model loaded from file, the input tensors
        `initial_state` and `constants` can be passed to `RNN.__call__` as part
        of `inputs` instead of by the dedicated keyword arguments. This method
        makes sure the arguments are separated and that `initial_state` and
        `constants` are lists of tensors (or None).
        # Arguments
        inputs: tensor or list/tuple of tensors
        initial_state: tensor or list of tensors or None
        constants: tensor or list of tensors or None
        # Returns
        inputs: tensor
        initial_state: list of tensors or None
        constants: list of tensors or None
        """
        if isinstance(inputs, list) and len(inputs) > 2:
            assert initial_state is None and constants is None
            if num_constants is not None:
                constants = inputs[-num_constants:]
                inputs = inputs[:-num_constants]
            initial_state = inputs[2:]
            inputs = inputs[:2]

        def to_list_or_none(x):
            if x is None or isinstance(x, list):
                return x
            if isinstance(x, tuple):
                return list(x)
            return [x]

        initial_state = to_list_or_none(initial_state)
        constants = to_list_or_none(constants)

        return inputs, initial_state, constants


    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        inputs, initial_state, constants = self._standardize_args(
            inputs, initial_state, constants, self._num_constants)

        if initial_state is None and constants is None:
            return super(Attentioncell, self).__call__(inputs, **kwargs)

        # If any of `initial_state` or `constants` are specified and are Keras
        # tensors, then add them to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            kwargs['initial_state'] = initial_state
            additional_inputs += initial_state
            self.state_spec = [InputSpec(shape=K.int_shape(state))
                               for state in initial_state]
            additional_specs += self.state_spec
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                                   for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec

        # at this point additional_inputs cannot be empty
        is_keras_tensor = K.is_keras_tensor(additional_inputs[0])
        for tensor in additional_inputs:
            if K.is_keras_tensor(tensor) != is_keras_tensor:
                raise ValueError('The initial state or constants of an ExternalAttentionRNNWrapper'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors'
                                 ' (a "Keras tensor" is a tensor that was'
                                 ' returned by a Keras layer, or by `Input`)')

        if is_keras_tensor:
            # Compute the full input spec, including state and constants
            full_input = inputs + additional_inputs
            #full_input = additional_inputs
            full_input_spec = self.input_spec + additional_specs
            #full_input_spec = additional_specs
            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(Attentioncell, self).__call__(full_input, **kwargs)
            self.input_spec = self.input_spec[:len(original_input_spec)]
            return output
        else:
            return super(Attentioncell, self).__call__(inputs, **kwargs)

    def get_constants(self, x):
        # add constants to speed up calculation
        constants = [x, K.dot(x, self._W1) + self._b2]
        return constants

    def get_config(self):
        config = {'return_attention': self.return_attention, 'weight_initializer': self.weight_initializer}
        base_config = super(Attentioncell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
################################################Citation#########################################################################
#             This implementation is not tested till now but if you found any modification I am always ready for changes        #
#                contact me at alok.rawat478@gmail.com                                                                          #
#                https://alokssingh.github.io/                                                                                  #
#https://github.com/zimmerrol/keras-utility-layer-collection/blob/e0a888277ee0121b88c8541ff7642e4615a76ce1/kulc/attention.py#L38#
#                                                                                                                               #
#https://github.com/alokssingh/imcap_keras/blob/master/imcap/layers/lstm_sent.py                                                #
#https://arxiv.org/pdf/1502.03044.pdfv                                                                                          #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#################################################################################################################################
