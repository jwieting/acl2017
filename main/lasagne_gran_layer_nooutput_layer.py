import lasagne
import numpy as np
import theano
from lasagne.layers import Gate
from lasagne import init
from lasagne import nonlinearities
from lasagne.utils import unroll_scan
from theano import tensor as T

class lasagne_gran_layer_nooutput_layer(lasagne.layers.MergeLayer):

    def __init__(self, incoming, num_units,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 #outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 avg_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=False,
                 mask_input=None,
                 only_return_final=False,
                 gran_type = 1,
                 **kwargs):

        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        self.model_type = gran_type
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, lasagne.layers.Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1
        if isinstance(cell_init, lasagne.layers.Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(lasagne_gran_layer_nooutput_layer, self).__init__(incomings, **kwargs)

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                         'forgetgate')

        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')

        #(self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         #self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units, ), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units, ), name="W_cell_to_forgetgate")

            #self.W_cell_to_outgate = self.add_param(
            #    outgate.W_cell, (num_units, ), name="W_cell_to_outgate")

        self.W_avg1 = self.add_param(
                init.Normal(0.1), (num_inputs, num_units), name="W_avg1")
        self.W_avg2 = self.add_param(
                init.Normal(0.1), (num_inputs, num_units), name="W_avg2")
        self.b_avg = self.add_param(
                init.Constant(0.), (num_units,), name="b_avg")

        if self.model_type == 3 or self.model_type == 4 or self.model_type == 5:
            self.W_avg12 = self.add_param(
                init.Normal(0.1), (num_inputs, num_units), name="W_avg12")
            self.W_avg22 = self.add_param(
                init.Normal(0.1), (num_inputs, num_units), name="W_avg22")
            self.b_avg2 = self.add_param(
                init.Constant(0.), (num_units,), name="b_avg2")
        if self.model_type == 4:
            self.W_avg3 = self.add_param(
                init.Normal(0.1), (num_inputs, num_units), name="W_avg3")
            self.W_avg32 = self.add_param(
                init.Normal(0.1), (num_inputs, num_units), name="W_avg32")

        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, lasagne.layers.Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, lasagne.layers.Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

        if isinstance(avg_init, lasagne.layers.Layer):
            self.avg_init = avg_init
        else:
            self.avg_init = self.add_param(
                avg_init, (1, self.num_units), name="avg_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        avg_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell], axis=1)

        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell], axis=0)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked

        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, avg_previous, *args):
            x=input_n
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            #outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            #if self.peepholes:
            #    outgate += cell*self.W_cell_to_outgate
            #outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = self.nonlinearity(cell)

            avg_input = T.dot(x, self.W_avg1) + T.dot(hid, self.W_avg2) + self.b_avg
            if self.model_type == 1:
                avg = x*nonlinearities.sigmoid(avg_input)
            elif self.model_type == 2:
                avg = hid*nonlinearities.sigmoid(avg_input)
            elif self.model_type == 3:
                avg_input2 = T.dot(x, self.W_avg12) + T.dot(hid, self.W_avg22) + self.b_avg2
                g1 = nonlinearities.sigmoid(avg_input)
                g2 = nonlinearities.sigmoid(avg_input2)
                avg = avg_previous*g1 + x*g2
            elif self.model_type == 4:
                avg_input = T.dot(x, self.W_avg1) + T.dot(hid, self.W_avg2) + T.dot(avg_previous, self.W_avg3) + self.b_avg
                avg_input2 = T.dot(x, self.W_avg12) + T.dot(hid, self.W_avg22) + T.dot(avg_previous, self.W_avg32) + self.b_avg2
                g1 = nonlinearities.sigmoid(avg_input)
                g2 = nonlinearities.sigmoid(avg_input2)
                avg = avg_previous*g1 + x*g2
            elif self.model_type == 5:
                avg_input2 = T.dot(x, self.W_avg12) + T.dot(hid, self.W_avg22) + self.b_avg2
                g1 = nonlinearities.sigmoid(avg_input)
                g2 = nonlinearities.sigmoid(avg_input2)
                avg = x*g1
                havg = hid*g2
                avg = avg + havg
            return [cell, hid, avg]

        def step_masked(input_n, mask_n, cell_previous, hid_previous, avg_previous, *args):
            cell, hid, avg = step(input_n, cell_previous, hid_previous, avg_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            avg = T.switch(mask_n, avg, avg_previous)

            return [cell, hid, avg]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, lasagne.layers.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, lasagne.layers.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        if not isinstance(self.avg_init, lasagne.layers.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            avg_init = T.dot(ones, self.avg_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked, self.W_avg1, self.W_avg2, self.b_avg]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.model_type == 3 or self.model_type == 5:
            non_seqs += [self.W_avg12, self.W_avg22, self.b_avg2]

        if self.model_type == 4:
            non_seqs += [self.W_avg12, self.W_avg22, self.b_avg2, self.W_avg3, self.W_avg32]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out, avg_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init, avg_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out, avg_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init, avg_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            avg_out = avg_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            avg_out = avg_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                avg_out = avg_out[:, ::-1]

        return avg_out
