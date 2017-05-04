import lasagne

class lasagne_average_layer(lasagne.layers.MergeLayer):
    
    def __init__(self, incoming, tosum = False, **kwargs):
        super(lasagne_average_layer, self).__init__(incoming, **kwargs)
        self.tosum = tosum

    def get_output_for(self, inputs, **kwargs):
        emb = inputs[0]
        mask = inputs[1]
        emb = (emb * mask[:, :, None]).sum(axis=1)
        if not self.tosum:
            emb = emb / mask.sum(axis=1)[:, None]
        return emb
    
    def get_output_shape_for(self, input_shape):
        #print input_shape
        return (input_shape[0][0],input_shape[0][2])
