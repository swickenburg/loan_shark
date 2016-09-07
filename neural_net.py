from IPython import display

def make_nn_dataset(
    X,
    y,
    len_feat_bool=False,
    ):
    num_classes = 2
    num_samples = len(y)
    len_feat = len(X[0])
    nn_dataset = ClassificationDataSet(len_feat, 1,
            nb_classes=num_classes)
    for i in range(num_samples):
        nn_dataset.addSample(X[i], y[i])

    nn_dataset._convertToOneOfMany()

    if len_feat_bool:
        return (nn_dataset, len_feat)
    else:
        return nn_dataset
    
    
(train, len_feat) = make_nn_dataset(X_train_cont, y_train_cont, len_feat_bool=True)
val = make_nn_dataset(X_val_cont, y_val_cont)


class DataSet:
    def __init__(self, train, val, len_feat):
        self.train = train
        self.val = val
        self.len_feat = len_feat
        
    def make_class_count_even(self):

        # i_even_gain = self.get_class_count_even_idx(self.y_gain_train_val)

        self.get_class_count_even_idx()

        # now make separate X arrays for gain and loss because of
        # different random split to make class counts even

        (self.X_train_val, self.y_train_val, self.value_train_val) = \
            (self.X_train_val[self.i_even],
             self.y_train_val[self.i_even],
             self.value_train_val[self.i_even])
            
ds = DataSet(train, val, len_feat)

class NeuralNet:

    def __init__(
        self,
        ds,
        hidden_layers=[100, 50, 25],
        num_classes=2,
        ):
        self.net = buildNetwork(
            ds.len_feat,
            hidden_layers[0],
            hidden_layers[1],
            num_classes,
            hiddenclass=SigmoidLayer,
            bias=True,
            outclass=SoftmaxLayer,
            )

        self.trainer = BackpropTrainer(
            self.net,
            dataset=ds.train,
            learningrate=0.0005,
            momentum=0.08,
            weightdecay=0.005,
            verbose=True,
            )

        # self.trainer = DeepBeliefTrainer

        self.ds = ds

    def train_net(self, num_epochs=500, title=''):

        # ds_train, val, gain_val, gain_cut,
        # err_train = []
        # err_val = []

        kappa_train = []
        kappa_val = []
        exp_gain = []
        y_train = self.ds.train['class'].flat[:].astype('int')
        y_val = self.ds.val['class'].flat[:].astype('int')
        net_list = []
        (fig, ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8, 10))
        for i in range(num_epochs):
            self.trainer.trainEpochs(1)
            net_list.append(self.net)

            # err_train.append(0.01*percentError(trainer.testOnClassData(),
            #                               train['class']))
            # err_val.append(0.01*percentError(trainer.testOnClassData(
            #                 dataset=val), val['class']))

            y_pred_val = \
                self.net.activateOnDataset(self.ds.val).argmax(axis=1)
            y_pred_train = \
                self.net.activateOnDataset(self.ds.train).argmax(axis=1)
            kappa_val.append(skll.metrics.kappa(y_val, y_pred_val))
            kappa_train.append(skll.metrics.kappa(y_train,
                               y_pred_train))

            # exp_gain.append(self.make_expected_gain(self.net, self.ds.val,
            #                                    self.ds.value_val, self.ds.y_cut))

            # plt.plot(np.array(err_train), '--b', lw=2)
            # plt.plot(np.array(err_val), '--r', lw=2)

            ax1.plot(np.array(kappa_train), c=my_map[1], lw=2)
            ax1.plot(np.array(kappa_val), c=my_map[0], lw=2)
            ax1.set_title(title)

            # ax2.plot(np.array(exp_gain), c=almost_black, lw=2)

            ax1.grid(True)
            display.clear_output(wait=True)
            display.display(plt.gcf())

        # trainEpochs(epochs=1
        # plot(errors[0])
        # plot(errors[1])
        # param for trainuntilconvergence(validationProportion=0.25)
        # y_pred_val = net.activateOnDataset(val).argmax(axis=1)
        # y_pred_train = net.activateOnDataset(train).argmax(axis=1)
        # print skll.metrics.kappa(y_val, y_pred_val)
        # return net_list, kappa_val, exp_gain
        
    def make_roc_curve(
        self,
        p_cut_list=np.arange(0., 1.01, 0.01),
        ):

        # p_pred_test = self.net.activateOnDataset(test)
        # y_pred_test = np.argmax(p_pred_test, axis=1)
        # p_pred_max_test = np.max(p_pred_test, axis=1)
        y_val = self.ds.val['class'].flat[:].astype('int')
        p_pos_test = self.net.activateOnDataset(self.ds.val)[:, 1]

        i_neg = np.where(y_val==0.)[0]
        i_pos = np.where(y_val==1.)[0]
        num_neg = float(len(y_val[i_neg]))
        num_pos = float(len(y_val[i_pos]))

        false_pos_rate = np.empty(len(p_cut_list))
        false_pos_rate.fill(np.nan)

        true_pos_rate = np.empty(len(p_cut_list))
        true_pos_rate.fill(np.nan)
        for (i, p_cut) in enumerate(p_cut_list):
        #     i_neg_cut = where(p_pred_max_test[i_neg] > p_cut)[0]
        #     i_pos_cut = where(p_pred_max_test[i_pos] > p_cut)[0]

            i_neg_cut = np.where(p_pos_test[i_neg] > p_cut)[0]
            i_pos_cut = np.where(p_pos_test[i_pos] > p_cut)[0]

        #     num_false_pos = sum(y_pred_test[i_neg[i_neg_cut]] == 1.)

            num_false_pos = len(i_neg[i_neg_cut])

        #     num_true_pos = sum(y_pred_test[i_pos[i_pos_cut]] == 1.)

            num_true_pos = len(i_pos[i_pos_cut])

            false_pos_rate[i] = num_false_pos / num_neg
            true_pos_rate[i] = num_true_pos / num_pos
        return (false_pos_rate, true_pos_rate, num_pos, num_neg)
    
net = NeuralNet(ds, [20,10,5])

net.train_net(num_epochs=200)

# pos_rate_list = []
pos_rate_list.append(net.make_roc_curve())

fig = figure(figsize=(8,8))
ax = fig.add_subplot(111, aspect='equal')
ax.plot(np.arange(0,1.01,0.1), np.arange(0,1.01,0.1),  c=dark_grey, lw=1)
for i in range(len(pos_rate_list)):
    if i == len(pos_rate_list)-1:
        ax.plot(pos_rate_list[i][0], pos_rate_list[i][1], 'o', c=almost_black)
    else:
        ax.plot(pos_rate_list[i][0], pos_rate_list[i][1], 'o', c=my_map[i], markeredgecolor=my_map[i])
xlabel('false pos rate', fontsize=14)
ylabel('true pos rate', fontsize=14)