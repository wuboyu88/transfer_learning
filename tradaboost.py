import numpy as np
from scipy.special import softmax
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


class TrAdaBoost:
    """
    Implement TrAdaBoost
    To read more about the TrAdaBoost, check the following paper:
    Dai, Wenyuan, et al. "Boosting for transfer learning." Proceedings of the 24th international conference on
    Machine learning. ACM, 2007.

    ps: Different from other implementations, use predict_proba instead of predict
    """

    def __init__(self, learner, num_iterations):
        self.learner = learner
        self.num_iterations = num_iterations
        self.beta_t = None
        self.models = []

    def fit(self, x_target, x_source, y_target, y_source):
        row_source = x_source.shape[0]
        row_target = x_target.shape[0]

        x_trans = np.concatenate((x_source, x_target), axis=0)
        y_trans = np.concatenate((y_source, y_target), axis=0)

        x_trans = np.asarray(x_trans)
        y_trans = np.asarray(y_trans)

        beta = 1 / (1 + np.sqrt(2 * np.log(row_source / self.num_iterations)))
        self.beta_t = np.zeros([self.num_iterations, 1])

        # 初始化权重
        weight_source = np.ones([row_source, 1]) / row_source
        weight_target = np.ones([row_target, 1]) / row_target
        weights = np.concatenate((weight_source, weight_target), axis=0)

        for i in range(self.num_iterations):
            sample_weights = self._calculate_weight(weights)
            self.learner.fit(x_trans, y_trans, sample_weights[:, 0])
            self.models.append(self.learner)
            result = self.learner.predict_proba(x_trans)[:, 1]
            error_rate = self._calculate_error_rate(y_target, result[row_source:row_source + row_target],
                                                    weights[row_source:row_source + row_target])

            if error_rate > 0.5:
                error_rate = 0.5
            if error_rate == 0:
                self.num_iterations = i
                break  # 防止过拟合

            self.beta_t[i] = error_rate / (1 - error_rate)

            # 调整源域样本权重
            for j in range(row_source):
                weights[j] = weights[j] * np.power(beta, (np.abs(result[j] - y_source[j])))

            # 调整目标域样本权重
            for j in range(row_target):
                weights[row_source + j] = weights[row_source + j] * np.power(self.beta_t[i],
                                                                             (-np.abs(
                                                                                 result[row_source + j] - y_target[j])))

    @staticmethod
    def _calculate_weight(weights):
        sum_weight = np.sum(weights)
        return np.asarray(weights / sum_weight)

    @staticmethod
    def _calculate_error_rate(y_target, y_predict, weight):
        total = np.sum(weight)
        return np.sum(weight[:, 0] / total * np.abs(y_target - y_predict))

    def predict(self, x_test):
        row_test = x_test.shape[0]
        result = np.ones([row_test, self.num_iterations])
        predict = np.ones([row_test, 1])
        for i in range(self.num_iterations):
            result[:, i] = self.models[i].predict_proba(x_test)[:, 1]
        for i in range(row_test):
            left = np.sum(result[i, int(np.ceil(self.num_iterations / 2)):self.num_iterations] * np.log(
                1 / self.beta_t[int(np.ceil(self.num_iterations / 2)):self.num_iterations]))
            right = 0.5 * np.sum(
                np.log(1 / self.beta_t[int(np.ceil(self.num_iterations / 2)):self.num_iterations]))
            if left >= right:
                predict[i] = 1
            else:
                predict[i] = 0
        return predict

    def predict_proba(self, x_test):
        """
        与其他实现不同的是加了softmax函数
        """
        row_test = x_test.shape[0]
        result = np.ones([row_test, self.num_iterations])
        predict = np.ones([row_test, 2])
        for i in range(self.num_iterations):
            result[:, i] = self.models[i].predict_proba(x_test)[:, 1]
        for i in range(row_test):
            left = np.sum(result[i, int(np.ceil(self.num_iterations / 2)):self.num_iterations] * np.log(
                1 / self.beta_t[int(np.ceil(self.num_iterations / 2)):self.num_iterations]))
            right = 0.5 * np.sum(
                np.log(1 / self.beta_t[int(np.ceil(self.num_iterations / 2)):self.num_iterations]))
            predict[i] = softmax([right, left])
        return predict


if __name__ == '__main__':
    # Generate data
    np.random.seed(0)
    # Generate training source data
    ns = 200
    ns_perclass = ns // 2
    mean_1 = (1, 1)
    var_1 = np.diag([1, 1])
    mean_2 = (3, 3)
    var_2 = np.diag([2, 2])
    Xs = np.r_[np.random.multivariate_normal(mean_1, var_1, size=ns_perclass),
               np.random.multivariate_normal(mean_2, var_2, size=ns_perclass)]
    ys = np.zeros(ns)
    ys[ns_perclass:] = 1
    # Generate training target data
    nt = 50
    # imbalanced
    nt_0 = nt // 10
    mean_1 = (6, 3)
    var_1 = np.diag([4, 1])
    mean_2 = (5, 5)
    var_2 = np.diag([1, 3])
    Xt = np.r_[np.random.multivariate_normal(mean_1, var_1, size=nt_0),
               np.random.multivariate_normal(mean_2, var_2, size=nt - nt_0)]
    yt = np.zeros(nt)
    yt[nt_0:] = 1
    # Generate testing target data
    nt_test = 1000
    nt_test_perclass = nt_test // 2
    Xt_test = np.r_[np.random.multivariate_normal(mean_1, var_1, size=nt_test_perclass),
                    np.random.multivariate_normal(mean_2, var_2, size=nt_test_perclass)]
    yt_test = np.zeros(nt_test)
    yt_test[nt_test_perclass:] = 1

    # transfer learning
    lightgbm_learner = lgb.LGBMClassifier(max_depth=3, n_estimators=100)
    trc = TrAdaBoost(learner=lightgbm_learner, num_iterations=5)
    trc.fit(Xt, Xs, yt, ys)
    print('lightgbm learner')
    print('train target auc: ', roc_auc_score(y_true=yt, y_score=trc.predict_proba(Xt)[:, 1]))
    print('test auc: ', roc_auc_score(y_true=yt_test, y_score=trc.predict_proba(Xt_test)[:, 1]))

    decision_tree_learner = DecisionTreeClassifier(max_depth=3)
    trc = TrAdaBoost(learner=decision_tree_learner, num_iterations=5)
    trc.fit(Xt, Xs, yt, ys)
    print('decision_tree learner')
    print('train target auc: ', roc_auc_score(y_true=yt, y_score=trc.predict_proba(Xt)[:, 1]))
    print('test auc: ', roc_auc_score(y_true=yt_test, y_score=trc.predict_proba(Xt_test)[:, 1]))
