#!/usr/bin/env python3
import sys
import os
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import traceback

sys.path.append('lib/')

# This is need for numpy to print the whole arrays (prob. dependent on numpy version)
np.set_printoptions(threshold=np.inf)

from helpers.experiment import Experiment
from helpers.params import get_argument_parser, get_params, print_params
from helpers.intt_builder import build_masked_graphs, build_unmasked_graph
from simulation.gen_input import INTTInputMasked, INTTInputUnmasked, BHatInfo, gen_skpv
from simulation.leak_data import LeakDataMasked, LeakDataUnmasked


def create_experiment_masked(params, rng, skpv, bhat):
    print("Generating masked inputs..")
    inttinput = INTTInputMasked.generate(
        rng, params['height'], params['kyber_k'], skpv, bhat
    )
    print("Simulating leakage..")
    leakdata = LeakDataMasked.generate(
        rng, inttinput, params['layers'], params['sigma']
    )
    g, mask_idx, skm_idx = build_masked_graphs(leakdata)
    r = Experiment.from_params(
        g,
        leakdata.maskshare.coeffs.tolist() + leakdata.skm.coeffs.tolist(),
        mask_idx + skm_idx,
        params,
        len(leakdata.zero_indices),
    )
    return r


def create_experiment_unmasked(params, rng, skpv, bhat):
    print("Generating secret key and ciphertext..")
    inttinput = INTTInputUnmasked.generate(
        rng, params['height'], params['kyber_k'], skpv, bhat
    )
    print("Simulating leakage..")
    leakdata = LeakDataUnmasked.generate(
        rng, inttinput, params['layers'], params['sigma']
    )
    g, idx = build_unmasked_graph(leakdata)
    r = Experiment.from_params(
        g, leakdata.sk.coeffs.tolist(), idx, params, len(leakdata.zero_indices)
    )
    return r


def avg_results(statslist):
    rank0 = sum([stats['rank_zero'] for stats in statslist]) / len(statslist)
    avg_rank = sum([stats['avg_rank'] for stats in statslist]) / len(statslist)
    success = sum([1 for stats in statslist if stats['success']]) / len(statslist)
    return rank0, avg_rank, success


def print_total_results(
    statslist, is_unmasked, number_coeffs, number_zero_indices, sigma
):
    rank0, avg_rank, success = avg_results(statslist)
    unknown_coeffs = number_coeffs - number_zero_indices
    rate = (rank0 - number_zero_indices) / unknown_coeffs
    print(
        "\nTotal (on average) with sigma {} and {} zeros in a{} setting:".format(
            sigma, number_zero_indices, "n unmasked" if is_unmasked else " masked"
        )
    )
    print("----------------------")
    print(
        "{} of {} coefficients of rank 0 ({} set to 0)".format(
            rank0, number_coeffs, number_zero_indices
        )
    )
    print(
        "{} of {} unknown coefficients with rank 0.".format(
            rank0 - number_zero_indices, unknown_coeffs
        )
    )
    print("Average rank is {}.".format(avg_rank))
    print("Average coefficients success rate: {}".format(rate))
    print("Success rate: {}".format(success))
    print("----------------------\n\n")
    return success, success * unknown_coeffs


def main():
    args = get_argument_parser().parse_args()
    params = get_params(args)
    print_params(args)

    total_results = {}  # will be saved for later plotting/analysis

    marker = itertools.cycle(('+', '.', 'o', '*'))

    if params['do_not_draw']:
        matplotlib.use('Agg')

    fig_suc, ax_suc = plt.subplots()
    fig_rec, ax_rec = plt.subplots()
    for bhat_info in params['bhat_infos']:
        total_results[bhat_info] = {}

        rates = []
        recs = []
        for sigma in params['sigmas']:
            total_results[bhat_info][sigma] = {}

            print("Sigma: ", sigma)
            print("==============")
            statslist = []
            resultslist = []
            # Terrible..
            params['sigma'] = sigma

            results = []
            for i_run in range(params['runs']):
                total_results[bhat_info][sigma][i_run] = {}
                seed = params['seed'] + i_run
                total_results[bhat_info][sigma][i_run]['seed'] = seed
                random.seed(seed)
                rng = np.random.default_rng(seed)

                try:
                    skpv_orig, skpv = gen_skpv(rng, params['height'], params['kyber_k'])
                    print("skpv: ", skpv)
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    bhat, len_zero_indices = bhat_info.generate_bhat(
                        rng, params['height'], params['kyber_k']
                    )
                    print("Bhat: ", bhat)
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    print(
                        "Experiment {} with seed {} and {} zeros:".format(
                            i_run, seed, len_zero_indices
                        )
                    )

                    if params['unmasked']:
                        r = create_experiment_unmasked(params, rng, skpv, bhat)
                    else:
                        r = create_experiment_masked(params, rng, skpv, bhat)
                    r.run(params['total_steps'])
                    results, stats, unrecovered = r.print_results()
                    resultslist.append(results)
                    statslist.append(stats)
                    print("")
                    np.savez_compressed(
                        "{}_exp_{}_{}_{}".format(
                            params['result_file'], bhat_info, sigma, i_run
                        ),
                        # r=r,
                        skpv=skpv,
                        bhat=bhat,
                        params=params,
                        results=results,
                        stats=stats,
                        unrecovered=unrecovered,
                    )
                except:
                    print("ERROR in run {} with sigma {}".format(i_run, sigma))
                    error = traceback.format_exc()
                    print(error)
                    total_results[bhat_info][sigma][i_run]['error'] = error
                else:
                    total_results[bhat_info][sigma][i_run]['skpv_orig'] = skpv_orig
                    total_results[bhat_info][sigma][i_run]['skpv'] = skpv
                    total_results[bhat_info][sigma][i_run]['bhat'] = bhat
                    total_results[bhat_info][sigma][i_run]['results'] = results
                    total_results[bhat_info][sigma][i_run]['stats'] = stats
                    total_results[bhat_info][sigma][i_run]['unrecovered'] = unrecovered

            if results:
                success_rate, rec = print_total_results(
                    statslist, params['unmasked'], len(results), len_zero_indices, sigma
                )
                rates.append((sigma, success_rate))
                recs.append((sigma, rec))
                total_results[bhat_info][sigma]['statslist'] = statslist
                total_results[bhat_info][sigma]['success_rate'] = success_rate
                total_results[bhat_info][sigma]['rec'] = rec
                total_results[bhat_info][sigma]['resultslist'] = resultslist

        label = str(len_zero_indices)
        if bhat_info.info_type == BHatInfo.BHatInfoType.GENERATED_BHAT:
            label += "({})".format(str(bhat_info.info))
        m = next(marker)
        ax_suc.plot(*zip(*rates), marker=m, label=len_zero_indices)
        ax_rec.plot(*zip(*recs), marker=m, label=len_zero_indices)
        total_results[bhat_info]['rates'] = rates
        total_results[bhat_info]['recs'] = recs

    ax_rec.legend()
    ax_rec.set_xticks(np.arange(0.0, max(params['sigmas']) + 0.2, 0.2))
    ax_rec.set_xlabel("sigma")
    ax_rec.set_ylabel("coefficients recovered")
    fig_rec.savefig(params['result_file'] + '_rec.png')

    ax_suc.legend()
    ax_suc.set_xticks(np.arange(0.0, max(params['sigmas']) + 0.2, 0.2))
    ax_suc.set_xlabel("sigma")
    ax_suc.set_ylabel("success rate")
    fig_suc.savefig(params['result_file'] + '_suc.png')

    np.savez_compressed(
        "{}".format(params['result_file']), total_results=total_results, params=params
    )


if __name__ == '__main__':
    main()
