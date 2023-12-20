import train
import data_process

loss_list = []
acc_list = []

result = train.train(loss_list, acc_list)
train.test(result, train.data)
data_process.plot_result(loss_list, acc_list)