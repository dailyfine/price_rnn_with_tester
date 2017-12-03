import tensorflow as tf
import numpy as np
import h5py
import os

import pandas as pd
import numpy as np
import functools

import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_path = "./tmp/model_"
max_length = 50
feature_number = 12

molit_start_time = '201401'
molit_end_time = '201601'

which_epoch_to_restore = 10
restore_session_flag = True
Training_flag = True
Test_flag = True
learning_rate = 0.01
n_hidden = 10
n_hidden2 = 128
n_hidden3 = 128

n_epoch = 100
n_input = feature_number
n_mini_batch_size = 25
n_decoder_input_feature = 10
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

def isConvertible_to_float(record):
    for i in record:
        try:
            np.asarray(i, np.float32)
        except:
            print("Error! it's not convertible to float!")
            return False
    return True

def make_batch(encoder_input, encoder_output):
    encoder_input_batch = []
    encoder_target_batch = []

    for batch_idx, ith_input_batch in enumerate(encoder_input):
        input_ = encoder_input[batch_idx]
        target_ = encoder_output[batch_idx]
        t_in = np.asarray(input_)
        t_out = np.asarray(target_)

        z_in = np.zeros_like(np.arange(max_length * feature_number).reshape(max_length, feature_number),
                             dtype=np.float32)
        z_out = np.zeros_like(np.arange(max_length), dtype=np.float32)

        if t_in.shape[0] <= max_length:
            z_in[:t_in.shape[0], :t_in.shape[1]] = t_in
            z_out[:t_out.shape[0]] = t_out
        else:
            z_in = t_in[-max_length:]
            z_out = t_out[-max_length:]

        encoder_input_batch.append(z_in)
        encoder_target_batch.append(z_out)

    return np.asarray(encoder_input_batch), np.asarray(encoder_target_batch)

def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)

def disjunction(*conditions):
    return functools.reduce(np.logical_or, conditions)

def molit_timecode_generator(start_code, end_code):
    result_timecode = []

    start_YY = int(start_code[:4])
    start_MM = int(start_code[4:])

    end_YY = int(end_code[:4])
    end_MM = int(end_code[4:])

    tmp_MM = start_MM
    tmp_YY = start_YY

    while tmp_YY * 100 + tmp_MM < end_YY * 100 + end_MM:
        temp_time = tmp_MM + tmp_YY * 100
        result_timecode.append(str(temp_time) + '_molit.csv')
        tmp_MM = tmp_MM + 1

        if tmp_MM >= 13:
            tmp_MM = 1
            tmp_YY = tmp_YY + 1

    return result_timecode



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

test_data = pd.read_csv('test.csv')

for idx_upper in range(len(test_data)):

    province = test_data.iloc[idx_upper]['시도']
    city = test_data.iloc[idx_upper]['시구군']
    village = test_data.iloc[idx_upper]['읍면동']
    number = test_data.iloc[idx_upper]['지번']
    name = test_data.iloc[idx_upper]['아파트명']
    size = test_data.iloc[idx_upper]['전용면적']
    result = pd.DataFrame()
    for item in molit_timecode_generator(molit_start_time, molit_end_time):
        # for item in molit_timecode_generator('201101', auction_time_code):
        # 이게 더 효율일지는 나중에 체크해 보도록 하자.
        data_molit = pd.read_csv('data_files/molit/' + item)
        c1 = data_molit['시도'] == province
        c2 = data_molit['시구군'] == city
        c3 = data_molit['읍면동'] == village
        c4 = data_molit['지번'] == number
        c5 = data_molit['아파트명'] == name
        c6 = data_molit['전용면적'] < (size + 10)
        c7 = data_molit['전용면적'] > (size - 10)

        # 레코드가 시/도/동 까지 같은 경우 지번이나 아프명 중에 어느 하나라도 일치하면 같은 매물로 여긴다.
        data_filtered = pd.DataFrame(data_molit[conjunction(c1, c2, c3, disjunction(c4, c5), c6, c7)])
        result = result.append(data_filtered)

    cleaned_encoder_sequence = result[['위도', '경도', '거래년도', '거래월', '건축년도', '층', '전용면적', '거래금액']]
    cleaned_encoder_sequence_np = cleaned_encoder_sequence.dropna(axis=0).as_matrix()

    if not result.empty and isConvertible_to_float(cleaned_encoder_sequence_np):
        encoder_input_batch = []
        encoder_output_batch = []

        for idx, item in enumerate(cleaned_encoder_sequence_np):
            #  ['위도', '경도', '거래년도', '거래월', '건축년도', '층', '전용면적', '거래금액']

            if idx < np.shape(cleaned_encoder_sequence_np)[0] - 1:
                latitude = cleaned_encoder_sequence_np[idx][0]
                longitude = cleaned_encoder_sequence_np[idx][1]
                current_apartment_selling_year = cleaned_encoder_sequence_np[idx][2]
                current_apartment_selling_month = cleaned_encoder_sequence_np[idx][3]
                current_apartment_constructed_year = cleaned_encoder_sequence_np[idx][4]
                current_apartment_floor = cleaned_encoder_sequence_np[idx][5]
                current_apartment_size = cleaned_encoder_sequence_np[idx][6]
                current_apartment_selling_price = cleaned_encoder_sequence_np[idx][7]

                predicting_apartment_selling_year = cleaned_encoder_sequence_np[idx + 1][2]
                predicting_apartment_selling_month = cleaned_encoder_sequence_np[idx + 1][3]
                predicting_apartment_floor = cleaned_encoder_sequence_np[idx + 1][5]
                predicting_apartment_size = cleaned_encoder_sequence_np[idx + 1][6]

                predicting_apartment_selling_price = cleaned_encoder_sequence_np[idx + 1][7]

                input_batch_pre = \
                    [latitude
                        , longitude
                        , current_apartment_selling_year
                        , current_apartment_selling_month
                        , current_apartment_floor
                        , current_apartment_size
                        , current_apartment_selling_price
                        , current_apartment_constructed_year
                        , predicting_apartment_selling_year
                        , predicting_apartment_selling_month
                        , predicting_apartment_floor
                        , predicting_apartment_size]

                encoder_input_batch.append(np.asarray(input_batch_pre))
                # encoder input 은 위도, 경도, 거래년도, 거래월, 층, 면적, 거래가격, 건축년도, 예측년도, 예측월, 예측 층, 예측 면적. 순이다.
                encoder_output_batch.append(predicting_apartment_selling_price)

            else:
                latitude = cleaned_encoder_sequence_np[idx][0]
                longitude = cleaned_encoder_sequence_np[idx][1]
                current_apartment_selling_year = cleaned_encoder_sequence_np[idx][2]
                current_apartment_selling_month = cleaned_encoder_sequence_np[idx][3]
                current_apartment_constructed_year = cleaned_encoder_sequence_np[idx][4]
                current_apartment_floor = cleaned_encoder_sequence_np[idx][5]
                current_apartment_size = cleaned_encoder_sequence_np[idx][6]
                current_apartment_selling_price = cleaned_encoder_sequence_np[idx][7]

                predicting_apartment_selling_year = test_data.iloc[idx_upper]['거래년도']
                predicting_apartment_selling_month = test_data.iloc[idx_upper]['거래월']
                predicting_apartment_floor = test_data.iloc[idx_upper]['층']
                predicting_apartment_size = test_data.iloc[idx_upper]['전용면적']

                predicting_apartment_selling_price = -100000000 #  테스트를 위한 가짜 값.

                input_batch_pre = \
                    [latitude
                        , longitude
                        , current_apartment_selling_year
                        , current_apartment_selling_month
                        , current_apartment_floor
                        , current_apartment_size
                        , current_apartment_selling_price
                        , current_apartment_constructed_year
                        , predicting_apartment_selling_year
                        , predicting_apartment_selling_month
                        , predicting_apartment_floor
                        , predicting_apartment_size]

                encoder_input_batch.append(np.asarray(input_batch_pre))
                # encoder input 은 위도, 경도, 거래년도, 거래월, 층, 면적, 거래가격, 건축년도, 예측년도, 예측월, 예측 층, 예측 면적. 순이다.
                encoder_output_batch.append(predicting_apartment_selling_price)

        if len(encoder_input_batch) > 50:
            ei_len = 50
        else:
            ei_len = len(encoder_input_batch)

        ei = [encoder_input_batch]
        eo = [encoder_output_batch]
        encoder_input_batch, encoder_target_batch = make_batch(ei, eo)
        encoder_input_batch = np.asarray(encoder_input_batch, dtype=np.float32)
        encoder_target_batch = np.asarray(encoder_target_batch, dtype=np.float32)

        for i in range(len(encoder_input_batch)):
            for j in range(max_length):
                # 201701년 기준으로 평균 및 표준 편차로 나누어줌
                encoder_input_batch[i][j][0] = (encoder_input_batch[i][j][0] - 36.4367) / 1.0547  # 위도
                encoder_input_batch[i][j][1] = (encoder_input_batch[i][j][1] - 127.5861) / 0.8790  # 경도
                encoder_input_batch[i][j][2] = (encoder_input_batch[i][j][
                                                    2] - 2014) / 2  # 거래년도 2011 - 2016 까지 평균 및 표준 편차
                encoder_input_batch[i][j][3] = (encoder_input_batch[i][j][3] - 6.5) / 3.45  # 거래월 2011 - 2016 까지 평균
                encoder_input_batch[i][j][4] = (encoder_input_batch[i][j][4] - 8.659) / 5.7851  # 층
                encoder_input_batch[i][j][5] = (encoder_input_batch[i][j][4] - 72.7554) / 24.3750  # 면적
                encoder_input_batch[i][j][6] = (encoder_input_batch[i][j][6] - 223586273) / 122304694  # 거래가격
                encoder_input_batch[i][j][7] = (encoder_input_batch[i][j][7] - 2000.1287) / 8.2695  # 건축년도
                encoder_input_batch[i][j][8] = (encoder_input_batch[i][j][
                                                    8] - 2014) / 2  # 예측년도 2011 - 2016 까지 평균 및 표준 편차
                encoder_input_batch[i][j][9] = (encoder_input_batch[i][j][9] - 6.5) / 3.45  # 예측월 2011 - 2016 까지 평균
                encoder_input_batch[i][j][10] = (encoder_input_batch[i][j][10] - 8.659) / 5.7851  # 예측 층
                encoder_input_batch[i][j][11] = (encoder_input_batch[i][j][11] - 72.7554) / 24.3750  # 면적

        encoder_target_batch = (encoder_target_batch - 223586273) / 122304694

        np_int = np.asarray([ei_len], dtype=np.int)
        encoder_prediction_= sess.run(encoder_result_tf,
                                                feed_dict={X: encoder_input_batch
                                                    # , Y: encoder_target_batch # 테스트 시에는 필요 없음
                                                    , lenXpl: np_int
                                                }
                                      )
        encoder_target_batch_ = encoder_target_batch * 122304694 + 223586273
        encoder_prediction_ = encoder_prediction_ * 122304694 + 223586273
        print(encoder_prediction_[0][ei_len - 1])
        test_data.iloc[idx_upper, test_data.columns.get_loc('거래금액')] = encoder_prediction_[0][ei_len - 1]
    else:
        print("we skipped this, since we have no data or unconvertable to float value.")
        print(province, city, village, number, name)
        print("------------------------------------------------------")

test_data.to_csv("blind_test_result.csv", sep=',', encoding='utf-8')