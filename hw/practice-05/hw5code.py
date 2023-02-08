import os

import numpy as np
from collections import Counter
from sklearn.base import ClassifierMixin

def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    feature_vector = np.array(feature_vector, dtype=np.float64)
    target_vector = np.array(target_vector, dtype=int)

    sort_indices = np.argsort(feature_vector)
    counter = Counter(feature_vector)

    Rl = np.cumsum(list(map(lambda x: x[1], sorted(counter.items(), key=lambda x: x[0]))))[:-1]

    unique_feature_vector, unique_indices = np.unique(feature_vector[sort_indices], return_index=True)
    target_vector = target_vector[sort_indices]

    thresholds = 0.5 * (unique_feature_vector[1:] + unique_feature_vector[:-1])

    cum_indices = (unique_indices-1)[1:]
    cum_target = np.cumsum(target_vector)[cum_indices]

    p1 = cum_target / Rl
    Hl = 1 - p1**2 - (1 - p1)**2

    cum_indices2 = (len(target_vector) - unique_indices - 1)[1:]
    cum_target2 = np.cumsum(target_vector[::-1])[cum_indices2]

    Rr = len(target_vector) - Rl
    p1 = cum_target2 / Rr
    Hr = 1 - p1**2 - (1 - p1)**2

    ginis = - Rl / len(target_vector) * Hl - Rr / len(target_vector) * Hr

    best_index = np.argmax(ginis)
    return np.array(thresholds), np.array(ginis), thresholds[best_index], ginis[best_index]


class DecisionTree(ClassifierMixin):
    def __init__(self, feature_types, max_depth=np.inf, min_samples_split=2, min_samples_leaf=3):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._depth = 0

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            self._depth = max(self._depth, depth)
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None

        #if True:
        if sub_X.shape[0] >= self._min_samples_split and depth < self._max_depth:
        #if sub_X.shape[0] > self._min_samples_leaf and depth < self._max_depth:
            for feature in range(sub_X.shape[1]):
                feature_type = self._feature_types[feature]
                categories_map = {}

                if feature_type == "real":
                    feature_vector = sub_X[:, feature]
                elif feature_type == "categorical":
                    counts = Counter(sub_X[:, feature])
                    clicks = Counter(sub_X[sub_y == 1, feature])
                    ratio = {}
                    for key, current_count in counts.items():
                        if key in clicks:
                            current_click = clicks[key]
                        else:
                            current_click = 0
                        ratio[key] = current_click / current_count
                    sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                    categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                    feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
                else:
                    raise ValueError

                if len(np.unique(feature_vector)) < 2:
                    continue

                _, _, threshold, gini = find_best_split(feature_vector, sub_y)
                if (gini_best is None or gini > gini_best) and sub_X[feature_vector < threshold].shape[0] >= self._min_samples_leaf and sub_X[feature_vector >= threshold].shape[0] >= self._min_samples_leaf:
                #if gini_best is None or gini > gini_best:
                    feature_best = feature
                    gini_best = gini
                    split = feature_vector < threshold

                    if feature_type == "real":
                        threshold_best = threshold
                    elif feature_type == "categorical":
                        threshold_best = list(map(lambda x: x[0],
                                                  filter(lambda x: x[1] < threshold, categories_map.items())))
                    else:
                        raise ValueError

        #if feature_best is None:
        if feature_best is None or sub_X.shape[0] < self._min_samples_split:
        #if feature_best is None or sub_X[split].shape[0] < self._min_samples_split or sub_X[~split].shape[0] < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            self._depth = max(self._depth, depth)
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        if len(sub_y[split]) == 0 or len(sub_y[np.logical_not(split)]) == 0:
            print("EMPTY!", len(sub_y[split]), len(sub_y[np.logical_not(split)]))
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth+1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth+1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        xj = x[feature]

        if self._feature_types[feature] == 'categorical':
            if np.isin(xj, node["categories_split"]):
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if xj < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)
        return self

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=False):
        return {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


if __name__ == "__main__":
    import os

    def run_test(input_path, out_path):
        thresholds, ginis, best_threshold, best_gini = [None] * 4
        with open(input_path, "r") as f:
            features_vec = np.array(list(map(float, f.readline().split(" "))))
            targets_vec = np.array(list(map(int, f.readline().split(" "))))

            thresholds, ginis, best_threshold, best_gini = find_best_split(features_vec, targets_vec)

        with open(out_path, "r") as f:
            thresholds_ans = np.array(list(map(float, f.readline().split(" "))))
            ginis_ans = np.array(list(map(float, f.readline().split(" "))))
            best_threshold_ans = float(f.readline())
            best_gini_ans = float(f.readline())

            assert np.allclose(thresholds, thresholds_ans), "{}\n{}".format(thresholds, thresholds_ans)
            assert np.allclose(ginis, ginis_ans), "{}\n{}".format(ginis, ginis_ans)
            assert np.allclose([best_threshold], [best_threshold_ans])
            assert np.allclose([best_gini], [best_gini_ans])

    def run_tests_A():
        tests_dir = "Tests/A"
        files = sorted(os.listdir(tests_dir))

        for i in range(0, len(files), 2):
            print("TEST %s" % (i//2 + 1))
            input_path = os.path.join(tests_dir, files[i])
            out_path = os.path.join(tests_dir, files[i+1])
            run_test(input_path, out_path)

    def run_test2(input_path, out_path):
        tree = None

        with open(input_path, "r") as f:
            feature_types = f.readline().strip().split(" ")

            tree = DecisionTree(feature_types=feature_types, min_samples_leaf=1, min_samples_split=1)

            vec = list(map(np.float64, f.readline().split(" ")))
            X_train = np.zeros((len(vec) // len(feature_types), len(feature_types)))

            for i in range(0, len(vec), len(feature_types)):
                X_train[i // len(feature_types), :] = vec[i: i+len(feature_types)]

            y = np.array(list(map(float, f.readline().strip().split(" "))))

            vec = list(map(np.float64, f.readline().split(" ")))
            X_test = np.zeros((len(vec) // len(feature_types), len(feature_types)))

            for i in range(0, len(vec), len(feature_types)):
                X_test[i // len(feature_types), :] = vec[i: i+len(feature_types)]

            tree.fit(X_train, y)
            y_pred = tree.predict(X_test)

        with open(out_path, "r") as f:
            y_true = np.array(list(map(float, f.readlines())))
            assert np.allclose(y_true, y_pred), "{}\n{}".format(y_true, y_pred)

    def run_tests_B():
        tests_dir = "Tests/B"
        files = sorted(os.listdir(tests_dir))

        for i in range(0, len(files), 2):
            print("TEST %s" % (i//2 + 1))
            input_path = os.path.join(tests_dir, files[i])
            out_path = os.path.join(tests_dir, files[i+1])
            run_test2(input_path, out_path)

    run_tests_B()