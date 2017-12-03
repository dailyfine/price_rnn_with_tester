import tensorflow as tf
import numpy as np
import h5py
import os
model_path = "./tmp/model_"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
max_length = 50
feature_number = 12

which_epoch_to_restore = 10
restore_session_flag = True
Training_flag = False
Test_flag = True
learning_rate = 0.01
n_hidden = 10
n_hidden2 = 128
n_hidden3 = 128

n_epoch = 100
n_input = feature_number
n_mini_batch_size = 1
n_decoder_input_feature = 10

# length returns lenX such that
# lenX[idx_specifies_batch] = the true length of times_steps
# i.e., maxlength - zero vectors which is filled in a make_batch method.
# This length info is very useful when calculating the cost removing the effect of zero vector's output.


def timecode_generator(start_code, end_code):
    result_timecode = []

    start_YY = int(start_code[:4])
    start_MM = int(start_code[4:])

    end_YY = int(end_code[:4])
    end_MM = int(end_code[4:])

    tmp_MM = start_MM
    tmp_YY = start_YY

    while tmp_YY * 100 + tmp_MM < end_YY * 100 + end_MM:
        temp_time = tmp_MM + tmp_YY * 100
        result_timecode.append(str(temp_time))
        tmp_MM = tmp_MM + 1

        if tmp_MM >= 13:
            tmp_MM = 1
            tmp_YY = tmp_YY + 1

    return result_timecode

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length_ = tf.reduce_sum(used, 1)
    length_ = tf.cast(length_, tf.int32)
    return length_

# np version of above method.
def np_length(sequence):
    used = np.sign(np.amax(np.abs(sequence), 2))
    length_ = np.sum(used, 1)
    return length_
# this make_batch method fills the input_batch with zero_vectors to make a fixed size tim_step batch element.

cost = tf.zeros_like(n_mini_batch_size)
with tf.device('/device:cpu:0'):
    # Declare the model variable and place holder.
    X = tf.placeholder(tf.float32, [None, max_length, n_input])
    Y = tf.placeholder(tf.float32, [None, max_length])
    lenXpl = tf.placeholder(tf.int32, [None])

    W1 = tf.Variable(tf.random_normal([n_hidden, n_hidden2]), name="W1")
    b1 = tf.Variable(tf.random_normal([n_hidden2]), name="b1")

    W2 = tf.Variable(tf.random_normal([n_hidden2, 1]), name="W2")
    b2 = tf.Variable(tf.random_normal([]), name="b2")

    cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

    # dropout is unnecessary.
    # cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
    multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
    outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32, sequence_length=length(X))

    # cost calculating using the tensors.
    encoder_cost_list = []
    encoder_result = []

    #  이걸 gpu 로 빠르게 학습 시키려면, mini_batch를 병렬적으로 연산을 해야할 것이다. <Numpy> 로 계산..
    #  그러나 지금은 일단 이렇게 for loop을 도는 것으로 하자.
    for b_idx in range(n_mini_batch_size):
        #  encoder has 2 layers.
        encoder_layer1_output = tf.matmul(outputs[b_idx], W1) + b1
        encoder_layer1_output = tf.nn.relu(encoder_layer1_output)
        encoder_layer2_output = tf.matmul(encoder_layer1_output, W2) + b2

        #  calculating encoder cost.
        z = np.zeros(max_length)  # same as np.zeros_like(max_length)
        len_ = lenXpl[b_idx]
        ones = tf.ones([len_], dtype=tf.float32) / tf.cast(len_, tf.float32)
        zeros = tf.zeros([max_length - len_])
        onezeros = tf.concat([ones, zeros], axis=0)

        meaningful_result = tf.square((encoder_layer2_output - tf.reshape(Y[b_idx], [tf.shape(Y[b_idx])[0], 1])))
        # L2 Norm 으로 변경 ...
        meaningful_result = tf.multiply(onezeros, meaningful_result)
        encoder_cost_list.append(tf.reduce_sum(meaningful_result))
        encoder_result.append(encoder_layer2_output)  # last time_step encoder prediction result to print

    encoder_result_tf = tf.stack(encoder_result)
    cost_ = tf.stack(encoder_cost_list)
    cost_result = tf.reduce_mean(cost_)

    training_loss = tf.summary.scalar('/training_loss', cost_result)
    validation_loss = tf.summary.scalar('/validation_loss', cost_result)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_result)
   # merged = tf.summary.merge_all()
    # run the session to calculate the result.

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

train_writer = tf.summary.FileWriter('result/hstrain', sess.graph)
test_writer = tf.summary.FileWriter('result/hstest', sess.graph)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

if restore_session_flag:
    try:
        print("successfully restored the previous session!")
        saver.restore(sess, model_path + str("444000"))
    except:
        print("There is no stored model. I'm creating one at ->", model_path)


training_data = timecode_generator('201601', '201604')
validation_data = timecode_generator('201610', '201612')

if Training_flag:
    # encoder input 은 위도, 경도, 거래년도, 거래월, 층, 면적, 거래가격, 건축년도, 예측년도, 예측월, 예측 층, 예측 면적. 순이다.

    for epoch in range(n_epoch):
        # random 하게, 시간을 고른다.

        if epoch % 100 == 0:
            s = saving_path = saver.save(sess, model_path+"_"+str(epoch))
            print('At epoch #'+str(epoch)+', model saving completed!')

        tr_randint = np.random.randint(0, len(training_data))
        h5f = h5py.File('batch_data_h5py/'+training_data[tr_randint]+'_np', 'r')

        batch_size_ = len(h5f['encoder_input_batch'][:])
        shuffled_batch_idx = np.arange(batch_size_)
        encoder_input_batch = h5f['encoder_input_batch'][:]
        encoder_length_vec = h5f['encoder_length_vector_batch'][:]
        encoder_length_vec = np.asarray(h5f['encoder_length_vector_batch'][:], dtype=np.int)
        encoder_target_batch = h5f['encoder_target_batch'][:]
        h5f.close()

        np.random.shuffle(shuffled_batch_idx)
        lenX = encoder_length_vec[shuffled_batch_idx][:n_mini_batch_size]
        np_lenX = np.asarray(lenX)
        _, loss, tr_summary = sess.run([optimizer, cost_result, training_loss],
                           feed_dict={X: encoder_input_batch[shuffled_batch_idx][:n_mini_batch_size]
                               ,Y: encoder_target_batch[shuffled_batch_idx][:n_mini_batch_size]
                               ,lenXpl: np_lenX
                             }
                           )
        # print('Epoch: ', '%04d' % (epoch + 1), 'training loss =', '{:.6f}'.format(loss))
        train_writer.add_summary(tr_summary, epoch)

        # Validation ======================================================================
        val_randint = np.random.randint(0, len(validation_data))
        h5f = h5py.File('batch_data_h5py/'+validation_data[val_randint]+'_np', 'r')

        batch_size_ = len(h5f['encoder_input_batch'][:])
        shuffled_batch_idx = np.arange(batch_size_)
        encoder_input_batch = h5f['encoder_input_batch'][:]
        encoder_length_vec = h5f['encoder_length_vector_batch'][:]
        encoder_length_vec = np.asarray(h5f['encoder_length_vector_batch'][:], dtype=np.int)
        encoder_target_batch = h5f['encoder_target_batch'][:]
        h5f.close()

        np.random.shuffle(shuffled_batch_idx)
        lenX = encoder_length_vec[shuffled_batch_idx][:n_mini_batch_size]
        np_lenX = np.asarray(lenX)
        loss, val_summary = sess.run([cost_result, validation_loss],
                           feed_dict={X: encoder_input_batch[shuffled_batch_idx][:n_mini_batch_size]
                               ,Y: encoder_target_batch[shuffled_batch_idx][:n_mini_batch_size]
                               ,lenXpl: np_lenX
                             }
                           )
        # print('Epoch:', '%04d' % (epoch + 1), 'validation loss =', '{:.6f}'.format(loss))
        train_writer.add_summary(val_summary, epoch)

    print('Optimization Complete!')


if Test_flag:

    h5f = h5py.File('batch_data_h5py/201611_np', 'r')
    batch_size_ = len(h5f['encoder_input_batch'][:])
    shuffled_batch_idx = np.arange(batch_size_)
    encoder_input_batch = h5f['encoder_input_batch'][:]
    encoder_length_vec = np.asarray(h5f['encoder_length_vector_batch'][:],dtype=np.int)
    encoder_target_batch = h5f['encoder_target_batch'][:]
    h5f.close()
    # encoder input 은 위도, 경도, 거래년도, 거래월, 층, 면적, 거래가격, 건축년도, 예측년도, 예측월, 예측 층, 예측 면적. 순이다.

    np.random.shuffle(shuffled_batch_idx)
    lenX = encoder_length_vec[shuffled_batch_idx][:n_mini_batch_size]
    encoder_prediction_ , summary = sess.run([encoder_result_tf, training_loss],
                                                        feed_dict={X: encoder_input_batch[shuffled_batch_idx][
                                                                      :n_mini_batch_size]
                                                            , Y: encoder_target_batch[shuffled_batch_idx][
                                                                      :n_mini_batch_size]
                                                            , lenXpl: lenX
                                                        }
                                  )
    encoder_prediction_ = encoder_prediction_ * 122304694 + 223586273
    encoder_result = []
    re = encoder_prediction_[:, :, 0]
    for i in range(n_mini_batch_size):
        tmp = [int(re[i][lenX[i] - 1]),((encoder_target_batch[shuffled_batch_idx][:n_mini_batch_size][i][lenX[i] - 1]) * 122304694 + 223586273)]
        encoder_result.append(tmp)
    result_np = np.asarray(encoder_result, dtype=np.float32)

    print(result_np)
    cost_of_test_result = np.mean(abs(result_np[:, 0] - result_np[:, 1]))
    print("encoder total cost: ", cost_of_test_result)

