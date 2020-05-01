#!/usr/bin/env python

""" 
    Illustrates the MAF algorithm (Papamakarios et al. 2018; Masked Autoregressive Flow for Density Estimation; https://arxiv.org/abs/1705.07057)
    Author: Lorne Whiteway.
"""


# Helper function for creating training data.
# Samples from a 2D gaussian distribution centred at (x_0, x_1) with indicated standard deviation (same in both dims) and no correlation.
# Returns an array of shape (num_samples, 2).
def gaussian_sample(num_data, x_0, x_1, std):
    import numpy as np
    return np.column_stack([np.random.normal(x_0, std, num_data), np.random.normal(x_1, std, num_data)])


def samples_from_three_gaussians(num_samples):
    import numpy as np
    return np.concatenate([gaussian_sample(num_samples//3, 1.0, 1.0, 0.3), gaussian_sample(num_samples//3, 2.0, 4.0, 0.6), gaussian_sample(num_samples//3, 4.0, 1.0, 0.7)])

# Mimics Fig 1 in the MAF paper.
def samples_from_C_shape(num_samples):
    import numpy as np
    x_1 = np.random.normal(0.0, 2.0, num_samples)
    x_0 = np.random.normal(0.25* x_1**2, 1.0, num_samples)
    return np.column_stack([x_0, x_1])


def samples_from_top_hat_stdev(num_samples):
    import numpy as np
    x_0 = np.random.normal(0.0, 1.0, num_samples)
    x_1 = np.random.normal(0.0, np.where(np.abs(x_0) < 0.75, 3.0, 1.0), num_samples)
    return np.column_stack([x_0, x_1])


# Can be amended as desired to accommodate alternative target distributions.
# Returns an array of shape (num_samples, 2).
def samples_from_target_distribution(num_samples):
    import numpy as np
    np.random.seed(1)
    
    # Can replace this if desired with one of the other 'samples_from...' functions to get a different target distribution.
    ret = samples_from_three_gaussians(num_samples)
    
    np.random.seed()
    return ret



# If you want to change the structure of the neural network then amend net_dimension and net_value. No other changes are needed.
# Ensure that the network has the autoregressive property!

# Return the number of adjustable weights in the neural network.
def net_dimension():
    return 28

# This routine defines the structure of the neural network.
# It returns output values as a function of net weights and input values (x). 
# net_weights is a vector of length net_dimension() containing the weights of the net
# x has shape (N,2) where each row is one pair of input x_0, x_1 values.
# Return value has shape (N,4); the columns are to be interpreted as: mean(p(x_0)), log(stdev)(p(x_0)), mean(p(x_1|x_0)), log(stdev)(p(x_1|x_0)).
def net_value(net_weights, x):

    import numpy as np
    import scipy.special as sp

    num_inputs = x.shape[0]

    # mean p(x_0)
    A = np.ones(num_inputs) * net_weights[0]

    # log(stdev) p(x_0)
    B = np.ones(num_inputs) * net_weights[1]

    # Values on the hidden layer (linear sigmoidal units).
    h_0 = sp.expit(np.outer(x[:,0], net_weights[2:8]) + net_weights[8:14])

    # mean p(x_1 | x_0)
    C = np.sum(h_0 * net_weights[14:20] + net_weights[20], axis=1)

    # log(stdev) p(x_1 | x_0)
    D = np.sum(h_0 * net_weights[21:27] + net_weights[27], axis=1)

    return np.column_stack([A, B, C, D])


 
# Suppress warning for log(0).
def safe_log(a):
    import numpy as np
    return np.where(a == 0.0, -10e10, np.log(a))



# Return value is the negative log probability of the rows of x as determined by the neural net with specified weights.
# net_weights should be a vector of length net_dimension()
# x should have shape (N,2).
# report_factor is a constant to multiply the loss function by. Use this (if desired) to get a loss function that converges to unity.
# Output is a vector of length N.
def neg_log_probability(net_weights, x, report_factor):
    import scipy.stats
    import numpy as np
    y = net_value(net_weights, x)
    nlp = report_factor * -np.sum(safe_log(scipy.stats.norm(y[:,0], np.exp(y[:,1])).pdf(x[:,0]) * scipy.stats.norm(y[:,2], np.exp(y[:,3])).pdf(x[:,1])))
    print(nlp)
    return nlp
    
    
# Train the neural net.
# initial_net_weights should be a vector of length net_dimension()
# training_data should have shape (N,2)
# report_factor is a constant to multiply the loss function by. Use this (if desired) to get a loss function that converges to unity.
# Return is a vector of length net_dimension() specifying the eights of the trained net.
def train_net(initial_net_weights, training_data, report_factor = 1.0):
    import scipy.optimize as spo
    
    print("Training...")
    res = spo.minimize(neg_log_probability, initial_net_weights, args=(training_data, report_factor))
    print(res.x)
    return res.x


# Sample from the distribution modelled by the net.
# net_weights should be a vector of length net_dimension() describing the net.
# Returns an array of shape (num_samples, 2).
# This is equation (3) in the MAF paper. Note we need two evaluations of the net.
def sample_from_modelled_distribution(net_weights, num_samples):
    import numpy as np
    import matplotlib.pyplot as plt

    in_x = np.zeros([num_samples, 2])
    y = net_value(net_weights, in_x)
    new_x_0 = np.random.normal(size = num_samples) * np.exp(y[:,1]) + y[:,0]
    in_x[:,0] = new_x_0
    y = net_value(net_weights, in_x)
    new_x_1 = np.random.normal(size = num_samples) * np.exp(y[:,3]) + y[:,2]

    return np.column_stack([new_x_0, new_x_1])


# Returns a vector of length net_dimension()
def get_initial_net_weights():
    import numpy as np
    # Adds random jitter to the starting point - I had been finding that if I started precisely at 0 then the optimiser tended to get stuck there.
    return np.random.uniform(-0.1, 0.1, net_dimension())


# Plot the trained net parameters: mean and log(std) of x_0 of p(x_0) and p(x_1|x_0)
def plot_net_parameters(net_weights, training_data):
    import numpy as np
    import matplotlib.pyplot as plt

    x_axis = np.linspace(np.amin(training_data[:,0]), np.amax(training_data[:,0]), 151)
    x = np.column_stack([x_axis, x_axis * 0.0])
    v = net_value(net_weights, x)

    labels = [r"$\mu(p(x_0))$", r"$\log(\sigma(p(x_0)))$", r"$\mu(p(x_1 \vert x_0))$", r"$\log(\sigma(p(x_1 \vert x_0)))$"]

    for i in range(4):
        plt.plot(x_axis, v[:,i], label=labels[i])
    plt.legend()
    plt.xlabel(r"$x_0$")
    plt.show()



# The samples from N(0, I) that would have been needed to be used in sample_from_modelled_distribution in order to reproduce x.
# net_weights should be a vector of length net_dimension() describing the net.
# x should have shape (N,2)
# Returns an array of shape (N,2)
# See equation (4) in the MAF paper.
def f_inverse(net_weights, x):

    import numpy as np

    v = net_value(net_weights, x)

    u_0 = (x[:,0] - v[:,0]) * np.exp(-v[:,1])
    u_1 = (x[:,1] - v[:,2]) * np.exp(-v[:,3])

    return np.column_stack([u_0, u_1])


# Plot a corner graph of the samples.
def plot_distribution(samples, axis_labels, title=""):
    from chainconsumer import ChainConsumer
    c = ChainConsumer()
    c.add_chain(samples, parameters=axis_labels, name=title)
    fig = c.plotter.plot(display=True, legend=True)



def rotate_2D(two_column_array, theta_in_radians):
    import numpy as np
    rotation_matrix = np.array([[np.cos(theta_in_radians), np.sin(theta_in_radians)], [-np.sin(theta_in_radians), np.cos(theta_in_radians)]])
    return np.dot(two_column_array, rotation_matrix)


def run():
    import numpy as np
    import matplotlib.pyplot as plt

    num_training_data = 7500
    num_dim = 2.0

    # Get training data
    training_data = samples_from_target_distribution(num_training_data) # This is what we actually use to train the net.

    # Number of times the MAF algorithm repeats the MADE algorithm.
    num_passes = 4

    for pass_number in range(num_passes):

        if True:
            plot_distribution(training_data, ["$x_1$", "$x_2$"], "Target distribution {}".format(pass_number))

        # Train the net
        report_factor = 1.0 / (num_training_data * num_dim * 0.5 * (np.log(2.0 * np.pi) + 1.0)) # See my notes p. 21. With this normalisation the loss function should converge to unity.
        trained_net_weights = train_net(get_initial_net_weights(), training_data, report_factor)

        if True:
            plot_net_parameters(trained_net_weights, training_data)

        if True:
            # Show the trained distribution
            data_from_trained_distribution = sample_from_modelled_distribution(trained_net_weights, num_training_data)
            plot_distribution(data_from_trained_distribution, ["$x_1$", "$x_2$"], "Trained distribution {}".format(pass_number))

        # Pull the training data back through f; agreement of this distribution with N(0, I) measures goodness-of-fit. 
        u = f_inverse(trained_net_weights, training_data)
        if True:
            plot_distribution(u, ["$u_1$", "$u_2$"], "U {}".format(pass_number))

        # Make the u values the training data for the next pass.
        training_data = u

        # Rotate the data (to offset the asymmetry in the algorithm's treatment of the two columns).
        # Use a rational number of radians as this ensures no periodicity.
        training_data = rotate_2D(training_data, 1.0)



if __name__ == '__main__':

    run()
