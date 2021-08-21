import numpy as np



class GFRM:


    def __init__(self, n_feature, d_activation_fn,  activation_fn, output_dim, learning_rate):

        self.n_feature = n_feature
        self.activation_n = activation_fn
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.d_activation_fn = d_activation_fn

        self.matrix = np.random.randn(self.n_feature, self.n_feature)

        self.weigths = [ np.random.randn(self.n_feature, self.n_feature) for _ in range(self.output_dim) ]
        self.weigths = np.array(self.weigths)
    
    def forward(self, data):

        if data.shape[1] == self.n_feature:
            maps = np.ones((data.shape[0], self.n_feature, self.n_feature))

            for i in range(data.shape[0]):
                for j in range(self.n_feature):
                    maps[i,j,j] = data[i,j]

            maps = self.matrix * maps

            maps = map(self.activation_n, maps)

            output_values = self.weigths * maps

            output_values = np.sum(output_values, axis=0)

            return output_values
                
        else:
            raise ValueError("Input data has not the same shape that numbers of features")
        pass

    def train(self, data):

        output_values = self.forward(data)
        pass 

    def apply_grad(self, gr_loss, loss):

        matrix_loss = np.zeros((self.n_feature, self.n_feature))

        for i in range(self.n_feature):
            for j in range(self.n_feature):
                
                for k in range(self.output_dim):
                    sum_weigth = self.weigths[k, i, j] / (np.sum(self.weigths[k]) - self.weigths[k, i, j])
                    matrix_loss[i, j] += sum_weigth * loss[k]


        d_matrix_loss = map(self.d_activation_fn, matrix_loss)

        self.matrix += self.learning_rate * d_matrix_loss

        for i, gr_l in enumerate(gr_loss):
            self.weigths[i] += self.learning_rate * gr_l
        