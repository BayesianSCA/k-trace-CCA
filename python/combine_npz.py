import numpy as np
import argparse
from test import *


def combine(files):
    file = files[0]
    data = np.load(file, allow_pickle=True)
    total_results = data['total_results'][()]
    params = data['params'][()]

    runs_total = 0
    for file in files[1:]:
        data = np.load(file, allow_pickle=True)
        total_results_new = data['total_results'][()]
        params_new = data['params'][()]

        # Check if files represent the same experiment
        # assert params['bhat_infos'] == params_new['bhat_infos'], "Npz does not correspond to same experiment!"
        assert (
            params['runs'] + params['seed'] <= params_new['seed']
        ), "!! SEED OVERLAPPING !!"
        # Remove multiple bhat runs
        if len(params['bhat_infos']) > 1:
            for param in params['bhat_infos'][1:]:
                total_results.pop(param)
            params['bhat_infos'] = [params['bhat_infos'][0]]

        if len(params_new['bhat_infos']) > 1:
            for param in params_new['bhat_infos'][1:]:
                total_results_new.pop(param)
            params_new['bhat_infos'] = [params_new['bhat_infos'][0]]

        # Store the additional results dicts
        intersection_sigma = list(
            np.intersect1d(params['sigmas'], params_new['sigmas'])
        )
        params['sigmas'] = list(np.unique(params['sigmas'] + (params_new['sigmas'])))
        params['intersection'] = intersection_sigma

        # see if values are not present in both sets...
        print("This is a possible error check if elements are in bot measurements...")

        for bhat_info, bhat_info_new in zip(
            params['bhat_infos'], params_new['bhat_infos']
        ):
            for sigma in params['sigmas']:
                if (sigma in total_results[bhat_info].keys()) and (
                    sigma in total_results_new[bhat_info_new].keys()
                ):
                    for new_run in range(params_new['runs']):
                        total_results[bhat_info][sigma][
                            new_run + params['runs']
                        ] = total_results_new[bhat_info_new][sigma][new_run]
                elif sigma in total_results_new[bhat_info_new].keys():
                    total_results[bhat_info][sigma] = total_results_new[bhat_info_new][
                        sigma
                    ]
                else:
                    pass

        # Calculate the new probabilites + recs

        runs = params['runs']
        runs_new = params_new['runs']

        # We have to store max amount of runs in a global variable
        # otherwise the script only works with one file
        runs_total = runs + runs_new

        rates = dict(total_results[bhat_info]['rates'])
        rates_new = dict(total_results_new[bhat_info_new]['rates'])
        recs = dict(total_results[bhat_info]['recs'])
        recs_new = dict(total_results_new[bhat_info_new]['recs'])
        for elem in intersection_sigma:
            rates[elem] = (rates[elem] * runs + rates_new[elem] * runs_new) / (
                runs + runs_new
            )
            recs[elem] = (recs[elem] * runs + recs[elem] * runs_new) / (runs + runs_new)

        total_results[bhat_info]['rates'] = list(rates.items())
        total_results[bhat_info]['recs'] = list(recs.items())

    params['runs'] = runs_total

    np.savez_compressed(
        "{}".format(params['result_file']), total_results=total_results, params=params
    )


def combine_same_amount(files):
    file = files[0]
    data = np.load(file, allow_pickle=True)
    total_results = data['total_results'][()]
    params = data['params'][()]

    runs_total = 0
    for file in files[1:]:
        data = np.load(file, allow_pickle=True)
        total_results_new = data['total_results'][()]
        params_new = data['params'][()]

        # Check if files represent the same experiment
        assert (
            params['bhat_infos'] == params_new['bhat_infos']
        ), "Npz does not correspond to same experiment!"
        # Remove multiple bhat runs
        if len(params['bhat_infos']) > 1:
            for param in params['bhat_infos'][1:]:
                total_results.pop(param)
            params['bhat_infos'] = [params['bhat_infos'][0]]

        if len(params_new['bhat_infos']) > 1:
            for param in params_new['bhat_infos'][1:]:
                total_results_new.pop(param)
            params_new['bhat_infos'] = [params_new['bhat_infos'][0]]

        # Store the additional results dicts
        params['sigmas'] = params['sigmas'] + params_new['sigmas']
        params['simgas'] = params['sigmas'].sort()

        for bhat_info in params['bhat_infos']:
            for sigma in params_new['sigmas']:
                total_results[bhat_info][sigma] = total_results_new[bhat_info][sigma]

            rates_new = (
                total_results[bhat_info]['rates']
                + total_results_new[bhat_info]['rates']
            )
            recs_new = (
                total_results[bhat_info]['recs'] + total_results_new[bhat_info]['recs']
            )
            rates_new.sort()
            recs_new.sort()
            total_results[bhat_info]['rates'] = rates_new
            total_results[bhat_info]['recs'] = recs_new

    np.savez_compressed(
        "{}".format(params['result_file']), total_results=total_results, params=params
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot the correlation during the key transfer.'
    )
    parser.add_argument(
        '-f',
        '--filenames',
        dest='filenames',
        nargs="+",
        type=str,
        required=True,
        help="Files to plot (<results_total>.npz)",
    )
    parser.add_argument(
        '--equal',
        action='store_true',
        help="Combine NPZ Files have the same amount of runs (default=false)",
    )
    args = parser.parse_args()

    if args.equal:
        combine_same_amount(args.filenames)
    else:
        combine(args.filenames)
