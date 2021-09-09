import os
import numpy as np
import matplotlib as mpl
import json

mpl.use('pgf')
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
from test import *

experiments = {}

## First block
experiments[
    "192-even-unmasked-0"
] = "-r 5 -n 192 --unmasked --seed 362989 -s 0.4 0.6 0.8 1.0"
experiments[
    "128-even-unmasked-0"
] = "-r 5 -n 128 --unmasked --seed 362989 -s 0.4 0.6 0.8 1.0"
experiments[
    "64-even-unmasked-0"
] = "-r 5 -n 64 --unmasked --seed 362989 -s 0.8 1.0 1.2 1.4"
experiments[
    "32-even0-unmasked-0"
] = "-r 5 -n 32 -k 4 --unmasked --seed 362989 -s 1.2 1.4 1.6 1.8"
experiments[
    "32-even1-unmasked-0"
] = "-r 5 -n 32 -k 4 --unmasked --seed 362989 -s 2.0 2.2 2.4 2.6"

experiments["192-even-masked-0"] = "-r 5 -n 192 --seed 362989 -s 0.4 0.6 0.8 1.0"
experiments["128-even-masked-0"] = "-r 5 -n 128 --seed 362989 -s 0.4 0.6 0.8 1.0"
experiments["64-even-masked-0"] = "-r 5 -n 64 --seed 362989 -s 0.8 1.0 1.2 1.4"
experiments["32-even0-masked-0"] = "-r 5 -n 32 -k 4 --seed 362989 -s 1.2 1.4 1.6 1.8"
experiments["32-even1-masked-0"] = "-r 5 -n 32 -k 4 --seed 362989 -s 2.0 2.2 2.4 2.6"

experiments[
    "192-odd-unmasked-0"
] = "-r 5 -n 192 --unmasked --seed 362989 -s 0.5 0.7 0.9 1.1"
experiments[
    "128-odd-unmasked-0"
] = "-r 5 -n 128 --unmasked --seed 362989 -s 0.5 0.7 0.9 1.1"
experiments[
    "64-odd-unmasked-0"
] = "-r 5 -n 64 --unmasked --seed 362989 -s 0.9 1.1 1.3 1.5"
experiments[
    "32-odd0-unmasked-0"
] = "-r 5 -n 32 -k 4 --unmasked --seed 362989 -s 1.3 1.5 1.7 1.9"
experiments[
    "32-odd1-unmasked-0"
] = "-r 5 -n 32 -k 4 --unmasked --seed 362989 -s 2.1 2.3 2.5 2.7"

experiments["192-odd-masked-0"] = "-r 5 -n 192 --seed 362989 -s 0.5 0.7 0.9 1.1"
experiments["128-odd-masked-0"] = "-r 5 -n 128 --seed 362989 -s 0.5 0.7 0.9 1.1"
experiments["64-odd-masked-0"] = "-r 5 -n 64 --seed 362989 -s 0.9 1.1 1.3 1.5"
experiments["32-odd0-masked-0"] = "-r 5 -n 32 -k 4 --seed 362989 -s 1.3 1.5 1.7 1.9"
experiments["32-odd1-masked-0"] = "-r 5 -n 32 -k 4 --seed 362989 -s 2.1 2.3 2.5 2.7"

## Second block
experiments[
    "192-even-unmasked-1"
] = "-r 15 -n 192 --unmasked --seed 362994 -s 0.4 0.6 0.8 1.0"
experiments[
    "128-even-unmasked-1"
] = "-r 15 -n 128 --unmasked --seed 362994 -s 0.4 0.6 0.8 1.0"
experiments[
    "64-even-unmasked-1"
] = "-r 15 -n 64 --unmasked --seed 362994 -s 0.8 1.0 1.2 1.4"
experiments[
    "32-even0-unmasked-1"
] = "-r 15 -n 32 -k 4 --unmasked --seed 362994 -s 1.2 1.4 1.6 1.8"
experiments[
    "32-even1-unmasked-1"
] = "-r 15 -n 32 -k 4 --unmasked --seed 362994 -s 2.0 2.2 2.4 2.6"

experiments["192-even-masked-1"] = "-r 15 -n 192 --seed 362994 -s 0.4 0.6 0.8 1.0"
experiments["128-even-masked-1"] = "-r 15 -n 128 --seed 362994 -s 0.4 0.6 0.8 1.0"
experiments["64-even-masked-1"] = "-r 15 -n 64 --seed 362994 -s 0.8 1.0 1.2 1.4"
experiments["32-even0-masked-1"] = "-r 15 -n 32 -k 4 --seed 362994 -s 1.2 1.4 1.6 1.8"
experiments["32-even1-masked-1"] = "-r 15 -n 32 -k 4 --seed 362994 -s 2.0 2.2 2.4 2.6"

experiments[
    "192-odd-unmasked-1"
] = "-r 15 -n 192 --unmasked --seed 362994 -s 0.5 0.7 0.9 1.1"
experiments[
    "128-odd-unmasked-1"
] = "-r 15 -n 128 --unmasked --seed 362994 -s 0.5 0.7 0.9 1.1"
experiments[
    "64-odd-unmasked-1"
] = "-r 15 -n 64 --unmasked --seed 362994 -s 0.9 1.1 1.3 1.5"
experiments[
    "32-odd0-unmasked-1"
] = "-r 15 -n 32 -k 4 --unmasked --seed 362994 -s 1.3 1.5 1.7 1.9"
experiments[
    "32-odd1-unmasked-1"
] = "-r 15 -n 32 -k 4 --unmasked --seed 362994 -s 2.1 2.3 2.5 2.7"

experiments["192-odd-masked-1"] = "-r 15 -n 192 --seed 362994 -s 0.5 0.7 0.9 1.1"
experiments["128-odd-masked-1"] = "-r 15 -n 128 --seed 362994 -s 0.5 0.7 0.9 1.1"
experiments["64-odd-masked-1"] = "-r 15 -n 64 --seed 362994 -s 0.9 1.1 1.3 1.5"
experiments["32-odd0-masked-1"] = "-r 15 -n 32 -k 4 --seed 362994 -s 1.3 1.5 1.7 1.9"
experiments["32-odd1-masked-1"] = "-r 15 -n 32 -k 4 --seed 362994 -s 2.1 2.3 2.5 2.7"

## Third block
experiments[
    "192-even-unmasked-2"
] = "-r 5 -n 192 --unmasked --seed 363009 -s 0.4 0.6 0.8 1.0"
experiments[
    "128-even-unmasked-2"
] = "-r 5 -n 128 --unmasked --seed 363009 -s 0.4 0.6 0.8 1.0"
experiments[
    "64-even-unmasked-2"
] = "-r 5 -n 64 --unmasked --seed 363009 -s 0.8 1.0 1.2 1.4"
experiments[
    "32-even0-unmasked-2"
] = "-r 5 -n 32 -k 4 --unmasked --seed 363009 -s 1.2 1.4 1.6 1.8"
experiments[
    "32-even1-unmasked-2"
] = "-r 5 -n 32 -k 4 --unmasked --seed 363009 -s 2.0 2.2 2.4 2.6"

experiments["192-even-masked-2"] = "-r 5 -n 192 --seed 363009 -s 0.4 0.6 0.8 1.0"
experiments["128-even-masked-2"] = "-r 5 -n 128 --seed 363009 -s 0.4 0.6 0.8 1.0"
experiments["64-even-masked-2"] = "-r 5 -n 64 --seed 363009 -s 0.8 1.0 1.2 1.4"
experiments["32-even0-masked-2"] = "-r 5 -n 32 -k 4 --seed 363009 -s 1.2 1.4 1.6 1.8"
experiments["32-even1-masked-2"] = "-r 5 -n 32 -k 4 --seed 363009 -s 2.0 2.2 2.4 2.6"

experiments[
    "192-odd-unmasked-2"
] = "-r 5 -n 192 --unmasked --seed 363009 -s 0.5 0.7 0.9 1.1"
experiments[
    "128-odd-unmasked-2"
] = "-r 5 -n 128 --unmasked --seed 363009 -s 0.5 0.7 0.9 1.1"
experiments[
    "64-odd-unmasked-2"
] = "-r 5 -n 64 --unmasked --seed 363009 -s 0.9 1.1 1.3 1.5"
experiments[
    "32-odd0-unmasked-2"
] = "-r 5 -n 32 -k 4 --unmasked --seed 363009 -s 1.3 1.5 1.7 1.9"
experiments[
    "32-odd1-unmasked-2"
] = "-r 5 -n 32 -k 4 --unmasked --seed 363009 -s 2.1 2.3 2.5 2.7"

experiments["192-odd-masked-2"] = "-r 5 -n 192 --seed 363009 -s 0.5 0.7 0.9 1.1"
experiments["128-odd-masked-2"] = "-r 5 -n 128 --seed 363009 -s 0.5 0.7 0.9 1.1"
experiments["64-odd-masked-2"] = "-r 5 -n 64 --seed 363009 -s 0.9 1.1 1.3 1.5"
experiments["32-odd0-masked-2"] = "-r 5 -n 32 -k 4 --seed 363009 -s 1.3 1.5 1.7 1.9"
experiments["32-odd1-masked-2"] = "-r 5 -n 32 -k 4 --seed 363009 -s 2.1 2.3 2.5 2.7"

## Forth block
experiments[
    "192-extra-unmasked-0"
] = "-r 5 -n 192 --unmasked --seed 42 -s 0.1 0.2 0.3 1.2 1.3 1.4"
experiments[
    "128-extra-unmasked-0"
] = "-r 5 -n 128 --unmasked --seed 42 -s 0.1 0.2 0.3 1.2 1.3 1.4"
experiments[
    "64-extra-unmasked-0"
] = "-r 5 -n 64 --unmasked --seed 42 -s 0.5 0.6 0.7 1.6 1.7 1.8"
experiments[
    "32-extra-unmasked-0"
] = "-r 5 -n 32 -k 4 --unmasked --seed 42 -s 0.9 1.0 1.1 2.8 2.9 3.0"

experiments["192-extra-masked-0"] = "-r 5 -n 192 --seed 42 -s 0.1 0.2 0.3 1.2 1.3 1.4"
experiments["128-extra-masked-0"] = "-r 5 -n 128 --seed 42 -s 0.1 0.2 0.3 1.2 1.3 1.4"
experiments["64-extra-masked-0"] = "-r 5 -n 64 --seed 42 -s 0.5 0.6 0.7 1.6 1.7 1.8"
experiments[
    "32-extra-masked-0"
] = "-r 5 -n 32 -k 4 --seed 42 -s 0.9 1.0 1.1 2.8 2.9 3.0"

## Results for 256 (taken from distributed graph - results identical)
sigma_list_256 = [0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
success_rate_list_256_unmasked = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.8,
    0.08,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
success_rate_list_256_masked = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.96,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
nbr_tries_list_256_unmasked = [5, 5, 5, 25, 25, 25, 25, 25, 25, 5, 5, 5]
nbr_tries_list_256_masked = [5, 5, 5, 25, 25, 25, 25, 25, 5, 5, 5, 5]


## Plot settings
marker = itertools.cycle(('o', 'v', "s", "D", ""))
colors = itertools.cycle(('#004b87', '#A8AD00', '#e87722', '#864e96', 'grey'))
dashed = itertools.cycle(('dashed', 'dashed', 'dashdot', 'dotted', 'solid'))
dashed_spacing = itertools.cycle(((5, 1), (4, 6)))  # , (5, 10), (5, 20)))
with open('python/plot/plt_config.json', "r") as configfile:
    config = json.load(configfile)
    plt.rcParams.update(config)


def conf_int(success_rate, nbr_tries):
    if nbr_tries > 5:
        confLevel = 0.95
        alpha = 1 - confLevel
        cum_prob = 1 - alpha / 2
        # z=probit(cum_prob)
        z = 1.96
        Wilson_min = (
            1 / (1 + z ** 2 / nbr_tries) * (success_rate + z ** 2 / (2 * nbr_tries))
            - z
            / (1 + z ** 2 / nbr_tries)
            * (
                success_rate * (1 - success_rate) / nbr_tries
                + z ** 2 / (4 * nbr_tries ** 2)
            )
            ** 0.5
        )
        Wilson_max = (
            1 / (1 + z ** 2 / nbr_tries) * (success_rate + z ** 2 / (2 * nbr_tries))
            + z
            / (1 + z ** 2 / nbr_tries)
            * (
                success_rate * (1 - success_rate) / nbr_tries
                + z ** 2 / (4 * nbr_tries ** 2)
            )
            ** 0.5
        )
        return Wilson_min, Wilson_max
    else:
        if success_rate != 1.0 and success_rate != 0.0:
            print(
                "WARNING: Not enough runs for unsure success rate: {}".format(
                    success_rate
                )
            )
        return success_rate, success_rate


# export_results = {
# 'unmasked': {
#     192 : {
#         0.5 : {
#             362989 : True,
#             362990 : True,
#             362991 : True,
#             362992 : False,
#             362993 : True,
#         }
#     }
# }
# }


class nesteddict(defaultdict):
    def __init__(self):
        super().__init__(nesteddict)


export_results = nesteddict()

for name, args in experiments.items():
    if not os.path.isfile('runs/{name}.tar.gz'.format(name=name)):
        print("Skipping {name}, tarball does not exist.".format(name=name))
        continue
    if not os.path.isdir('runs/extracted'):
        os.mkdir('runs/extracted')
    if not os.path.isdir('runs/extracted/{name}'.format(name=name)):
        os.mkdir('runs/extracted/{name}'.format(name=name))
        if (
            os.system(
                "tar -xzf runs/{name}.tar.gz -C runs/extracted/{name}".format(name=name)
            )
            != 0
        ):
            raise OSError("Could not extract tarball {name}".format(name=name))
    print(name)
    print(args)
    data = np.load(
        "runs/extracted/{name}/results.npz".format(name=name), allow_pickle=True
    )
    total_results = data['total_results'][()]
    params = data['params'][()]
    # print(params)
    # print(total_results)
    masking = 'unmasked' if params['unmasked'] else 'masked'
    for bhat_info, bhat_results in total_results.items():
        nbr_nonzeros = bhat_info.get_nbr_nonzeros(params['height'], params['kyber_k'])
        for sigma, sigma_results in bhat_results.items():
            if not isinstance(sigma, float):
                continue
            for run, run_results in sigma_results.items():
                if not isinstance(run, int):
                    continue
                seed = run_results['seed']
                success = run_results['stats']['success']
                export_results[masking][nbr_nonzeros][sigma][seed] = success
# print(export_results)

# Add merged results for 256 nonzeros
export_results['masked'][256]['merged'] = True
export_results['masked'][256]['sigma_list'] = sigma_list_256
export_results['masked'][256]['success_rate_list'] = success_rate_list_256_masked
export_results['masked'][256]['nbr_tries_list'] = nbr_tries_list_256_masked
export_results['unmasked'][256]['merged'] = True
export_results['unmasked'][256]['sigma_list'] = sigma_list_256
export_results['unmasked'][256]['success_rate_list'] = success_rate_list_256_unmasked
export_results['unmasked'][256]['nbr_tries_list'] = nbr_tries_list_256_unmasked

for masking, mask_results in export_results.items():
    # generate figure
    # plt.style.use('tableau-colorblind10')
    fig_suc, ax_suc = plt.subplots()
    for nbr_nonzeros, nbr_nz_results in sorted(mask_results.items(), reverse=True):
        print("#" * 80)
        print(
            "{mask} - {n}".format(
                mask=masking,
                n=nbr_nonzeros,
            )
        )
        sigma_list = [0.0]
        success_rate_list = [1.0]
        confidence_min_list = [1.0]
        confidence_max_list = [1.0]
        if (
            'merged' in nbr_nz_results
        ):  # special case for 256, as we already have the combined results
            sigma_list += nbr_nz_results['sigma_list']
            success_rate_list += nbr_nz_results['success_rate_list']
            for i, success_rate in enumerate(nbr_nz_results['success_rate_list']):
                nbr_tries = nbr_nz_results['nbr_tries_list'][i]
                confidence_min, confidence_max = conf_int(success_rate, nbr_tries)
                confidence_min_list.append(confidence_min)
                confidence_max_list.append(confidence_max)
        else:
            for sigma, sigma_results in sorted(nbr_nz_results.items()):
                nbr_tries = len(sigma_results)
                success_rate = sum(sigma_results.values()) / nbr_tries
                confidence_min, confidence_max = conf_int(success_rate, nbr_tries)
                sigma_list.append(sigma)
                success_rate_list.append(success_rate)
                confidence_min_list.append(confidence_min)
                confidence_max_list.append(confidence_max)
                print(
                    "{sigma}: {min} - {val} - {max}".format(
                        sigma=sigma,
                        min=confidence_min,
                        max=confidence_max,
                        val=success_rate,
                    )
                )
        sigma_list.append(3.2)
        success_rate_list.append(0.0)
        confidence_min_list.append(0.0)
        confidence_max_list.append(0.0)
        # plot success_rate
        m = next(marker)
        d = next(dashed)
        color_tmp = next(colors)
        if d != 'dashed':
            ax_suc.plot(
                sigma_list,
                success_rate_list,
                linestyle=d,
                color=color_tmp,
                label='{}'.format(nbr_nonzeros),
            )
        else:
            d_spacing = next(dashed_spacing)
            ax_suc.plot(
                sigma_list,
                success_rate_list,
                linestyle=d,
                dashes=d_spacing,
                color=color_tmp,
                label='{}'.format(nbr_nonzeros),
            )
        ax_suc.fill_between(
            sigma_list,
            confidence_min_list,
            confidence_max_list,
            alpha=0.1,
            color=color_tmp,
        )
    # finish plot

    x_axis_step = 4
    x_axis_ticks = list(np.arange(0.0, 3.1 + 0.2, 0.1))
    x_axis_label = [
        round(x_axis_ticks[i], 1) if i % x_axis_step == 0 else ''
        for i in range(len(x_axis_ticks))
    ]

    y_axis_step = 2

    y_axis_ticks = list(np.arange(0, ax_suc.dataLim.y1 + 0.1, 0.1))
    y_axis_labels = [
        round(y_axis_ticks[i], 1) if i % y_axis_step == 0 else ''
        for i in range(len(y_axis_ticks))
    ]

    ax_suc.legend()
    ax_suc.set_xlim(0, 3.2)
    ax_suc.set_xticks(x_axis_ticks)
    ax_suc.set_xticklabels(x_axis_label)
    ax_suc.set_yticks(y_axis_ticks)
    ax_suc.set_yticklabels(y_axis_labels)
    ax_suc.set_xlabel("$\sigma$")
    ax_suc.set_ylabel("success rate")
    # plt.rc('grid', linewidth=0.6, alpha=0.3)
    # plt.grid(True)
    fig_suc.tight_layout(pad=0.05, rect=(-0.01, -0.03, 1, 1))
    fig_suc.savefig(masking + '_blocked_results.pdf')
