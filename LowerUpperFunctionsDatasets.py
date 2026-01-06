import numpy as np
from itertools import product

def triangular_membership(x, a, b, c):
    x = np.asarray(x)
    mu = np.zeros_like(x, dtype=float)

    left = (a <= x) & (x <= b)
    mu[left] = (x[left] - a) / (b - a)

    right = (b < x) & (x <= c)
    mu[right] = (c - x[right]) / (c - b)

    return mu


def precompute_memberships(df, feature_partitions):
    memberships = {}  # klíč: (feature, term_index) -> mu-vektor

    for feature, params in feature_partitions.items():
        x = df[feature].values
        for idx, (a, b, c) in enumerate(params):
            memberships[(feature, idx)] = triangular_membership(x, a, b, c)

    return memberships


def compute_F_A_up_for_all_combinations(df, feature_partitions, f_values):
    """
    feature_partitions: dict
        např. {"X1": [(a,b,c),...], "X2": [(a,b,c),...]}
    f_values: 1D array, např. df["Y"].values
    """

    memberships = precompute_memberships(df, feature_partitions)

    features = list(feature_partitions.keys())
    term_ranges = [range(len(feature_partitions[feat])) for feat in features]

    results = []

    for comb in product(*term_ranges):
        # comb je např. (i1, i2, ..., im)
        mu_list = []
        for feat, term_idx in zip(features, comb):
            mu_list.append(memberships[(feat, term_idx)])

        # T-norma přes dimenze: minimum
        mu_A = np.minimum.reduce(mu_list)

        # vlastní funkcionál F_A^up
        products = mu_A * f_values
        F_A_up = np.max(products) if np.any(mu_A > 0) else 0.0

        results.append({
            "terms": {feat: idx for feat, idx in zip(features, comb)},
            "F_A_up": F_A_up
        })

    return results
def mu_A_of_x(x_point, feature_partitions):
    mu_dict = {}
    for feat, params in feature_partitions.items():
        mu_dict[feat] = []
        for (a,b,c) in params:
            mu_dict[feat].append(triangular_membership(np.array([x_point[feat]]), a, b, c)[0])
    return mu_dict
def f_A_upT_multi(x_point, feature_partitions, results):
    """
    x_point: dict, např. {"X1": 0.3, "X2": 4.1}
    feature_partitions: definice fuzzy trianglů
    results: výstupy z compute_F_A_up_for_all_combinations(...)
    """
    mu_dict = mu_A_of_x(x_point, feature_partitions)

    ratios = []

    for res in results:
        # poskládej μ_A(x) přes dimenze
        mu_values = []
        for feat, term_idx in res["terms"].items():
            mu_values.append(mu_dict[feat][term_idx])

        mu_A_x = np.min(mu_values)   # nebo prod etc.

        if mu_A_x > 0:
            ratios.append(res["F_A_up"] / mu_A_x)
        else:
            ratios.append(np.inf)

    return np.min(ratios)

