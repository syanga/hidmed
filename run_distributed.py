import argparse
import numpy as np
from hidmed import *
from tqdm import tqdm
import pickle

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


estimators = [
    ProximalInverseProbWeighting,
    ProximalOutcomeRegression,
    ProximalMultiplyRobust,
]


job_lookup = {
    0: (1, "a", 2), # good
    1: (5, "a", 1), # almost there but coverage goes down
    2: (5, "b", 6), # ok, but coverage goes down as n goes up?
    3: (1, "b", 1), # good
    4: (5, "c", 2), # bias goes up but fine
    5: (1, "c", 1), # starts ok but gets worse?
}


# sample_sizes = (np.array([1250, 2500, 5000]) * 3 // 4).astype(int)
# sample_sizes = (np.array([250, 500, 1000])).astype(int
sample_sizes = (np.array([125, 250, 500])).astype(int)
folds = 5


if __name__ == "__main__":
    # determine which experiment to run -- 600 in total
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", help="Which job", type=int, default=0)
    args = parser.parse_args()

    # run each job 100 times
    job_type = args.job // 100
    job_run = args.job % 100
    dim, setup, seed = job_lookup[job_type]

    # save results here
    filename = f"assets/results_{job_type}_{job_run}.pkl"
    results = {}

    # problem dimensions
    if dim == 5:
        xdim, zdim, wdim, mdim, udim = 5, 2, 2, 2, 1
    else:
        xdim, zdim, wdim, mdim, udim = 1, 1, 1, 1, 1

    # set up model
    datagen = LinearHidMedDGP(
        xdim, zdim, wdim, mdim, udim, setup=setup, seed=seed
    )
    true_psi = datagen.true_psi()
    dataset_full = datagen.sample_dataset(sample_sizes[-1], seed=job_run)

    print(f"Initialized Job {job_type}, run {job_run}.")

    # run experiments
    for n in sample_sizes:
        # get dataset of sample size n
        dataset = dataset_full[:n]

        # hyperparameter tuning: save fitted parameters, do not perform estimation
        try:
            tuned_param_dict = pickle.load(open(f"assets/tuned_params_{dim}_{setup}_{n}", "rb"))
        except:
            print(f">>> Tuning {n} samples, {folds}-fold, setup {setup}, dim {dim}")
#            param_dict = {
#                i: {"q": {"gamma2": 1e-2}, "h": {"gamma2": 1e-2}}
#                for i in range(folds)
#            }
            tuner = ProximalMultiplyRobust("c", folds=folds, num_runs=400)#, param_dict=param_dict)
            tuner.fit(dataset)
            tuned_param_dict = tuner.param_dict
            pickle.dump(tuned_param_dict, open(f"assets/tuned_params_{dim}_{setup}_{n}", "wb"))
            continue

        # estimation
        for estimator in estimators:
            print(f">>> {n} samples, {folds}-fold, setup {setup}, dim {dim}, est: {estimator.__name__}")

            predictor = estimator(
                setup, folds=folds, param_dict=tuned_param_dict, verbose=False,
            )
            predictor.fit(dataset)
            estimate = predictor.evaluate(reduce=False)

            res = {}
            res["true_psi"] = true_psi
            res["estimate"] = np.mean(estimate.flatten())
            res["bias"] = np.mean(estimate.flatten()) - true_psi
            res["mse"] = calculate_mse(estimate.flatten(), true_psi)
            res["anb"] = absolute_normalized_bias(estimate.flatten(), true_psi)
            res["covered"] = is_covered(estimate.flatten(), true_psi)
            res["ci_width"] = confidence_interval(estimate.flatten())

            print("pred:", res["estimate"])
            print("true:", true_psi)
            print("bias:", res["bias"])
            print("mse:", res["mse"])
            print("ci_width", res["ci_width"])
            print("is_covered", res["covered"])

            # bootstrap CI for PMR only
            if estimator.__name__ == "ProximalMultiplyRobust":
                psi_means = []
                for run in tqdm(range(100)):
                    predictor = estimator(
                        setup, folds=folds, param_dict=tuned_param_dict, verbose=False
                    )
                    predictor.fit(dataset)
                    psi_means.append(predictor.evaluate(reduce=True))

                bootstrap_ci = [np.quantile(psi_means, 0.05), np.quantile(psi_means, 0.95)]
                inside = bootstrap_ci[0] <= true_psi and true_psi <= bootstrap_ci[1]
                res["bootstrap_ci"] = np.array([bootstrap_ci[0], bootstrap_ci[1]])
                res["bootstrap_ci_width"] = bootstrap_ci[1] - bootstrap_ci[0]
                res["bootstrap_covered"] = inside

                print("bootstrap covered:", res["bootstrap_covered"])

            # save results
            results[dim, setup, n, estimator.__name__,job_run] = res

    pickle.dump(results, open(filename, "wb"))
