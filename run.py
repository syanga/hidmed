import argparse
import numpy as np
from hidmed import *
from tqdm import tqdm
import pickle

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


job_lookup = {
    0: (1, "a", 2),
    1: (5, "a", 1),
    2: (1, "b", 3),
    3: (5, "b", 4),
    4: (1, "c", 5),
    5: (5, "c", 6),
}

sample_sizes = (np.array([1250, 2500, 5000])).astype(int)

estimators = [
    ProximalInverseProbWeighting,
    ProximalOutcomeRegression,
    ProximalMultiplyRobust,
]

num_evals = 100
folds = 5


if __name__ == "__main__":
    # determine which experiment to run
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", help="Which job", type=int, default=0)
    args = parser.parse_args()
    dim, setup, seed =job_lookup[args.job]

    # save results here
    filename = f"assets/results_{setup}_{dim}.pkl"

    # problem dimensions
    if dim == 5:
        xdim, zdim, wdim, mdim, udim = 5, 2, 2, 2, 1
    else:
        xdim, zdim, wdim, mdim, udim = 1, 1, 1, 1, 1

    # load any existing file
    try:
        results = pickle.load(open(filename, "rb"))
    except:
        results = {}

    print("problem setup complete! Existing results:")
    for key in results.keys():
        print(key)

    # run experiments
    for n in sample_sizes:
        # set up model
        datagen = LinearHidMedDGP(
            xdim, zdim, wdim, mdim, udim, setup=setup, seed=seed
        )
        true_psi = datagen.true_psi()

        # hyperparameter tuning
        tuner = ProximalMultiplyRobust("c", folds=folds, num_runs=400)
        dataset = datagen.sample_dataset(n, seed=seed+1+np.random.choice(num_evals))
        tuner.fit(dataset, seed=np.random.randint(2**32))
        tuned_param_dict = tuner.param_dict

        # estimation
        for estimator in estimators:
            print(f">>> {n} samples, {folds}-fold, setup {setup}, dim {dim}, est: {estimator.__name__}")
            if (dim, setup, n, estimator.__name__) in results.keys():
                continue

            print(
                f"Running {estimator.__name__} for {dim}-dimensional case, n={n}, setup={setup}"
            )

            # evaluation
            res = {
                "true_psi": np.zeros(num_evals),
                "estimates": [],
                "estimate": np.zeros(num_evals),
                "bias": np.zeros(num_evals),
                "mse": np.zeros(num_evals),
                "anb": np.zeros(num_evals),
                "covered": np.zeros(num_evals),
                "ci_width": np.zeros(num_evals),
                "bootstrap_ci": np.zeros((num_evals, 2)),
                "bootstrap_ci_width": np.zeros(num_evals), 
                "bootstrap_covered": np.zeros(num_evals),
            }
            for idx, i in enumerate(tqdm(range(seed + 1, seed + num_evals + 1))):
                dataset_i = datagen.sample_dataset(n, seed=i)
                predictor = estimator(
                    setup, folds=folds, param_dict=tuned_param_dict, verbose=False,
                )
                predictor.fit(dataset_i, seed=i)
                estimate = predictor.evaluate(reduce=False)

                res["estimates"].append(estimate)
                res["true_psi"][idx] = true_psi
                res["estimate"][idx] = np.mean(estimate.flatten())
                res["bias"][idx] = np.mean(estimate.flatten()) - true_psi
                res["mse"][idx] = calculate_mse(estimate.flatten(), true_psi)
                res["anb"][idx] = absolute_normalized_bias(estimate.flatten(), true_psi)
                res["covered"][idx] = is_covered(estimate.flatten(), true_psi)
                res["ci_width"][idx] = confidence_interval(estimate.flatten())

                # bootstrap CI for PMR only
                if estimator.__name__ == "ProximalMultiplyRobust":
                    psi_means = []
                    for run in range(100):
                        predictor = estimator(
                            setup, folds=folds, param_dict=tuned_param_dict, verbose=False
                        )
                        predictor.fit(dataset_i, seed = num_evals+1+run)
                        psi_means.append(predictor.evaluate(reduce=True))

                    bootstrap_ci = [np.quantile(psi_means, 0.05), np.quantile(psi_means, 0.95)]
                    inside = bootstrap_ci[0] <= true_psi and true_psi <= bootstrap_ci[1]
                    res["bootstrap_ci"][idx, 0] = bootstrap_ci[0]
                    res["bootstrap_ci"][idx, 1] = bootstrap_ci[1]
                    res["bootstrap_ci_width"] = bootstrap_ci[1] - bootstrap_ci[0]
                    res["bootstrap_covered"][idx] = inside

            results[dim, setup, n, estimator.__name__] = res
            print("bias", np.mean(res["bias"]))
            print("mse", np.mean(res["mse"]))
            print("est", np.mean(res["estimate"]))
            print("true", true_psi)
            print("ci_width", np.mean(res["ci_width"]))
            print("coverage", np.mean(res["covered"]))

            if estimator.__name__ == "ProximalMultiplyRobust":
                print("bootstrap coverage", np.mean(res["bootstrap_covered"]))

            pickle.dump(results, open(filename, "wb"))