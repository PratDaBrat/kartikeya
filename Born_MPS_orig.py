import numpy as np
import tensorflow as tf
import itertools
import tensorflow.contrib.eager as tfe
from scipy import stats
import sys
import os
import copy
from numpy.random import choice
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from spsa import SimultaneousPerturbationOptimizer


tfe.enable_eager_execution()

np.random.seed(44)
tf.compat.v1.set_random_seed(1)


def pixel_to_matrix(pixel, matrix_dim=2):
    matrix = np.zeros((matrix_dim, matrix_dim,))
    matrix[int(pixel), int(pixel)] = 1.
    return matrix

def numpy_one_hot(dataset, input_size, legs):
    l = []
    for i in range(dataset.shape[0]):
        a = dataset[i]
        b = np.zeros((input_size, legs))
        b[np.arange(input_size), a] = 1
        l.append(b)
    return np.array(l)

def amp_to_prob(amp, epsilon):
    prob = []
    for i in range(len(amp)):
        if amp[i] != 0:
            prob.append(amp[i] / sum(amp))
        else:
            prob.append(epsilon)
    return prob


def kl_divergence(p, q):
    eps = 0.0000001
    p = amp_to_prob(p, eps)
    q = amp_to_prob(q, eps)
    kld = np.sum([p[i] * np.log(p[i] / q[i]) for i in range(len(q))])
    return kld


def MPS(bonds, input_size, legs):
    cells = []
    # first cell
    first_cell = tf.compat.v1.get_variable(name='cell_id0', dtype=tf.float64,
                                 shape=(legs,bonds[0]), initializer=tf.compat.v1.random_uniform_initializer(0, 1))
    cells.append(first_cell)
    for i in range(1, input_size - 1):
        cell = tf.compat.v1.get_variable(name='cell_id' + str(i), dtype=tf.float64,
                               shape=( legs,bonds[i - 1], bonds[i]), initializer=tf.compat.v1.random_uniform_initializer(0, 1))
        cells.append(cell)
    # last cell
    i = input_size - 2
    last_cell = tf.compat.v1.get_variable(name='cell_id' + str(i + 1), dtype=tf.float64,
                                shape=(bonds[i],legs), initializer=tf.compat.v1.random_uniform_initializer(0, 1))

    cells.append(last_cell)
    return cells


def get_identity_operator(input_size, legs):
    operators = []
    for i in range(input_size):
        temp = tf.eye(legs, dtype=tf.float64)
        operators.append(temp)
    return operators

def operator_contraction_efficient(operator, bonds, cells, legs):
    assert (cells[0].get_shape().as_list()[0] == legs)

    assert (len(cells) == len(operator))

    top_cells = cells
    bot_cells = []

    for i in range(len(cells)):
        if i==len(cells)-1:
            temp = tf.tensordot(operator[i], cells[i], [0, 1])
        else:
            temp = tf.tensordot(operator[i], cells[i], [0, 0])
        bot_cells.append(temp)

    for i in range(len(cells)):
        try:
            if i ==len(cells)-1:
                contractions = tf.tensordot(contractions, bot_cells[i], [0, -1])
                contractions = tf.tensordot(contractions, top_cells[i], [[0, 1], [0, 1]])
            else:
                contractions = tf.tensordot(contractions, bot_cells[i], [0, -2])
                contractions = tf.tensordot(contractions, top_cells[i], [[0, 1], [-2, 0]])
        except:
            assert (i == 0)
            contractions = bot_cells[0]
            contractions = tf.tensordot(contractions, top_cells[i], [0, 0])
    return contractions


def layers_contraction(input_data, cells,measured_qubits):
    all_psi = []

    for image in input_data:
        contracted_cells = copy.copy(cells)
        psi=0
        for i,q in enumerate(measured_qubits):
            pixel = image[i]
            cell = cells[q]
            if q==max_qubits-1:
                temp = tf.tensordot(pixel, cell, [0, -1])
            else:
                temp = tf.tensordot(pixel, cell, [0, 0])
            contracted_cells[q]=temp

        for i in range(len(contracted_cells) - 1):
            try:
                if len(contracted_cells[i + 1].shape)==1:             # ---> it is just for when last qubit is measured. can be written nicer if the for loop is incremented reverse.
                    psi = tf.tensordot(psi, contracted_cells[i + 1], [-1, -1])
                else:
                    psi = tf.tensordot(psi, contracted_cells[i + 1], [-1, -2])
            except:
                assert (i == 0)
                psi = tf.tensordot(contracted_cells[i], contracted_cells[i + 1], [-1, -2])
        psi_squared=get_psi_squared(psi)
        all_psi.append(psi_squared)
    return all_psi


def get_psi_squared(psi):
    axes = list(np.arange((len(psi.shape))))
    psi_sqrt = tf.tensordot(psi, psi, [axes, axes])
    return psi_sqrt


def loss_function(bonds, input_data, legs, cells,measured_qubits,epsilon):
    with tfe.GradientTape() as tape:
        identity_operator = get_identity_operator(len(cells), legs)
        z_tensor = operator_contraction_efficient(identity_operator, bonds, cells, legs)

        psi_2 = layers_contraction(input_data, cells,measured_qubits)
        loss = -(tf.reduce_sum(input_tensor=tf.math.log(psi_2) + epsilon) / len(psi_2)) + tf.math.log(z_tensor + epsilon)
    grads = tape.gradient(loss, cells)
    return loss, grads

def get_segment_operator(generated_outputs, max_qubits,legs):
    layer_tmp = []
    for tensor_ix in range(max_qubits):
        id_tensor = tf.eye(legs, dtype=tf.float64)
        layer_tmp.append(id_tensor)

    for k,v in generated_outputs.items():
        obs_matrix = pixel_to_matrix(v, matrix_dim=legs)
        obs_tensor = tf.constant(obs_matrix, dtype=tf.float64)
        layer_tmp[k]=obs_tensor
    return layer_tmp


def partial_probability(generated_outputs, max_qubits, cells,legs):

    segment_obs = get_segment_operator(generated_outputs, max_qubits,legs)
    segment_prob_unnormalized = operator_contraction_efficient(segment_obs,bonds, cells,legs)
    return segment_prob_unnormalized

def generate_sample(cells, measured_qubits):
    generated_outputs = {}

    for site_ix in measured_qubits:

        generated_outputs[site_ix]=0
        p_0= partial_probability(generated_outputs, max_qubits, cells,legs)

        generated_outputs[site_ix]=1
        p_1= partial_probability(generated_outputs, max_qubits, cells,legs)
        # Make it conditional prob.
        epsilon = 1e-100
        cond_p_0 = p_0 + epsilon
        cond_p_1 = p_1 + epsilon
        norm_prob = (cond_p_0 + cond_p_1)       #---> check if they are not 1
        cond_p_0 = cond_p_0 / norm_prob
        cond_p_1 = cond_p_1 / norm_prob

        # Sample new pixel.
        custm = stats.rv_discrete(name='custm', values=([0, 1],[cond_p_0, cond_p_1]))
        pixel_val = custm.rvs(size=1, random_state=np.random)

        generated_outputs[site_ix] = pixel_val[0]

    return list(generated_outputs.values())

def data_size_fixer(dataset, max_qubits,measured_qubits,legs):
    l = []
    for i in range(dataset.shape[0]):
        a=dataset[i]
        # print(a)
        # print(measured_qubits)
        temp=np.array(['I']*max_qubits)
        temp[measured_qubits]=a
        b = np.zeros((max_qubits, legs))
        b[np.array(measured_qubits), a] = 1
        l.append(b)
        # print(b)
        # exit()
    return np.array(l)

if __name__ == '__main__':

    res_dir = 'results/GHZ_data_Stablizer_fix'
    checkpoints_dir = 'checkpoints/checkpoints_GHZ_data_Stablizer_fix'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    mode = sys.argv[1]
    noise = sys.argv[2]
    m_val = int(sys.argv[3])
    m_val_prev=int(m_val)-1


    batch_size = 100
    legs = 2
    max_qubits=8

    NO_samples = 1000
    bond_dim = 8
    tot_epoch=10
    tot_epoch_prev=10

    measurement_info_dic={'0':[0,1],'1':[1,2],'2':[2,3],'3':[3,4],'4':[4,5],'5':[5,6],'6':[6,7],'7':[0,1,2,3,4,5,6,7]}
    file = np.load("mydata/GHZ_data_Stablizer_fix/ghz_q8_DP" + str(noise) + "m" + str(m_val) + "_sample.npz",allow_pickle=True)
    measured_qubits=measurement_info_dic[str(m_val)]

    train = file['input3']

    input_size=int(train.shape[-1])

    np.random.shuffle(train)
    train_data = numpy_one_hot(train.astype(int), input_size, 2)

    bonds = [bond_dim] * (max_qubits - 1)
    assert (len(bonds) == max_qubits - 1)
    cells = MPS(bonds, max_qubits, legs)
    
    print(len(cells))
    
    exit()
    saver = tfe.Saver(cells)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=5e-2)
#     optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-2,decay=0.5,momentum=0.5)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=5e-2)
    global_step = tf.compat.v1.train.get_or_create_global_step()

    ##############################
    if mode == 'train':
        if m_val_prev!=-1:
            print("==================================")
            print('loading the previous mps trained over measured qubits %s with m_val %s'%(measurement_info_dic[str(m_val_prev)],m_val_prev))
            print("==================================")
            epoch_to_load_prev=str(tot_epoch_prev-1)
            saver.restore(checkpoints_dir + "/checkpoint_q8_D" + str(noise) + "_bonds" + str(bond_dim) + "_m" + str(m_val_prev) + "_sample.ckpt-" + epoch_to_load_prev)

        print('measured_qubits: %s'%(measured_qubits))
        output_file = open(res_dir + "/Loss_q8_noiseD" + str(noise) + "_bonds" + str(bond_dim) + "_m" + str(m_val) + "_sample.txt",'w')
        NO_batches = int(train_data.shape[0] / batch_size)

        for epoch in range(tot_epoch):
            for it in range(NO_batches):
                loss_value, grads = loss_function(bonds, train_data[it * batch_size:(it + 1) * batch_size], legs, cells,measured_qubits,epsilon=1e-5)
                grads_and_vars = zip(grads, cells)
                optimizer.apply_gradients(grads_and_vars, global_step)

                print("Epoch {:1d}: it {:03d}: Loss: {:.3f}".format(epoch, it, loss_value))
                output_file.write("{:03d} {:.3f}".format((NO_batches * epoch) + it, loss_value) + '\n')

        saver.save(checkpoints_dir + "/checkpoint_q8_D" + str(noise) + "_bonds" + str(bond_dim) + "_m" + str(m_val) + "_sample.ckpt", global_step=epoch)

        samples=[]
        for i in range(NO_samples):
            if i%500==0:
                print(i)
            sample=generate_sample(cells,measured_qubits)
            samples.append(sample)
        np.savez(res_dir+"/q8_noiseD"+str(noise)+"epoch"+str(tot_epoch-1)+"_bonds"+str(bond_dim)+"_m" + str(m_val)+"_sample.npz",
                 samples=samples)

    elif mode == 'sample':
        epoch_to_load = str(tot_epoch-1)

        saver = tfe.Saver(cells)
        saver.restore(checkpoints_dir + "/checkpoint_q8_D" + str(noise) + "_bonds" + str(bond_dim) + "_m" + str(m_val) + "_sample.ckpt-" + epoch_to_load)
        samples=[]

#         for i in range(NO_samples):
#             if i%500==0:
#                 print(i)
#             sample=generate_sample(cells,measured_qubits)
#             samples.append(sample)
#         np.savez(res_dir+"/q8_noiseD"+str(noise)+"epoch"+str(epoch_to_load)+"_bonds"+str(bond_dim)+"_m" + str(m_val)+"_sample.npz",
#                  samples=samples)

    elif mode == 'plot':

        epoch_to_load = str(tot_epoch-1)
        all_config = [tuple(i) for i in itertools.product([0, 1], repeat=input_size)]

        ######## plot samples from cirq
        file = np.load("mydata/GHZ_data_Stablizer_fix/ghz_q8_DP" + str(noise) + "m" + str(m_val) + "_sample.npz",allow_pickle=True)
        output1_DP0 = file['input3']

        output3_DP0 = [tuple(i) for i in output1_DP0]
        dic1_DP0 = {}
        for i in output3_DP0:
            try:
                dic1_DP0[i] += 1
            except:
                dic1_DP0[i] = 1

        counts1_DP0 = [(k, dic1_DP0[k]) if k in dic1_DP0 else (k, 0) for k in all_config]
        amplitude = [e[1] for e in counts1_DP0]

        file = np.load(
            res_dir + "/q8_noiseD" + str(noise) + "epoch" + str(epoch_to_load) + "_bonds" + str(bond_dim) + "_m" + str(m_val) + "_sample.npz",allow_pickle=True)
        output_gen1_DP0 = file['samples']
        output_gen3_DP0 = [tuple(i) for i in output_gen1_DP0]

        dic_gen1_DP0 = {}
        for i in output_gen3_DP0:
            try:
                dic_gen1_DP0[i] += 1
            except:
                dic_gen1_DP0[i] = 1
        counts_gen1_DP0 = [(k, dic_gen1_DP0[k]) if k in dic_gen1_DP0 else (k, 0) for k in all_config]

        x = np.arange(2 ** input_size)
        amplitude_gen = [e[1] for e in counts_gen1_DP0]
        width = 1
        f, ax = plt.subplots(figsize=(18, 5))
        rects1 = ax.bar(x - width / 2, amplitude, width, color='red', label='Circuit Outcome')
        rects2 = ax.bar(x + width / 2, amplitude_gen, width, color='c', label='Generated by MPS')

        kl_val=kl_divergence(amplitude,amplitude_gen)
        plt.xlabel("Configuration")
        plt.ylabel("Outcome count")
        plt.title('q8_noise=' + str(noise) + ', D=' + str(bond_dim) + ', m=' + str(m_val) + ', kl={:.2f}'.format(kl_val))
        plt.savefig('mps_samples_cirq_q8_noise' + str(noise) + '_bonds' + str(bond_dim) + '_epochs' + str(
            epoch_to_load) + '_m' + str(m_val) + '_sample.png')

    else:
        print("+++++++++++++++++++")
        print("+++++++++++++++++++")
        print("+++++++++++++++++++")
        print("+++++++++++++++++++")
        print("wrong mode")
        exit()









