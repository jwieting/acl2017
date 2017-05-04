import lasagne

class lasagne_sum_layer(lasagne.layers.MergeLayer):
    
    def __init__(self, incoming, **kwargs):
        super(lasagne_sum_layer, self).__init__(incoming, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        emb1 = inputs[0]
        emb2 = inputs[1]
        return emb1 + emb2
    
    def get_output_shape_for(self, input_shape):
        return input_shape[0]
