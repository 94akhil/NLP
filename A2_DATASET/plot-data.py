import matplotlib.pyplot as plt
epochs = [1, 2, 3, 4, 5, 6]
# training_accuracy = [0.42825,0.459125,0.45425,0.461,0.460875,0.482125,0.4445,0.428,0.450125 ]
#     # [0.470375,0.56475,0.592375,0.619375,0.63125,0.652375]
#
# validation_accuracy = [0.44875,0.46125,0.40375,0.44875,0.44625,0.4575,0.44,0.46625,0.45125]
#     # [0.47875,0.5675,0.5825,0.5875,0.60125,0.5975]
#
# plt.figure(figsize=(10, 5))
# plt.plot(epochs, training_accuracy, label='Training Accuracy')
# plt.plot(epochs, validation_accuracy, label='Validation Accuracy', linestyle='--')
# plt.title('Training and Validation Accuracy over Epochs For H-dim = 6')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.show()

loss =[1.2194989919662476,1.225028157234192,1.1024401187896729,0.874223530292511,1.1206896305084229,0.8016642332077026]
# loss = [0.06811816990375519,0.06960661709308624,0.06071237847208977,0.12340398132801056,0.06602360308170319,0.050269607454538345,0.0486891008913517,0.05264333635568619,0.08155739307403564]
plt.figure(figsize=(10, 5))
plt.plot(epochs, loss, label='Loss',color='red')
plt.title('Loss Over Epochs For H-dim = 10')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()