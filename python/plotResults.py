import numpy as np
import argparse
import matplotlib as mpl
import json

mpl.use('pgf')
import matplotlib.pyplot as plt
from test import *
import itertools

marker = itertools.cycle(('o', 'v', "s", "D", ""))
colors = itertools.cycle(('#004b87', '#A8AD00', '#e87722', '#864e96', 'grey'))
dashed = itertools.cycle(('dashed', 'dashed', 'dashdot', 'dotted', 'solid'))
dashed_spacing = itertools.cycle(((5, 1), (4, 6), (5, 10), (5, 20)))


def plot(filenames, nr_zeros):
    # plt.style.use('tableau-colorblind10')
    fig_suc, ax_suc = plt.subplots()
    fig_rec, ax_rec = plt.subplots()

    for path, nrzero in zip(filenames, nr_zeros):

        data = np.load(path, allow_pickle=True)

        total_results = data['total_results'][()]

        params = data['params'][()]
        if len(params['bhat_infos']) > 1:
            res_rates = [
                (0.0, 0.0)
                for _ in range(len(total_results[params['bhat_infos'][0]]['rates']))
            ]
            res_recs = [
                (0.0, 0.0)
                for _ in range(len(total_results[params['bhat_infos'][0]]['recs']))
            ]
            for bhat in params['bhat_infos']:
                for i in range(len(total_results[bhat]['rates'])):
                    res_rates[i] = tuple(
                        map(sum, zip(res_rates[i], total_results[bhat]['rates'][i]))
                    )
                    res_recs[i] = tuple(
                        map(sum, zip(res_recs[i], total_results[bhat]['recs'][i]))
                    )

            rates = []
            recs = []
            for rate, rec in zip(res_rates, res_recs):
                rates.append(
                    tuple([rate[i] / len(params['bhat_infos']) for i in range(0, 2)])
                )
                recs.append(
                    tuple([rec[i] / len(params['bhat_infos']) for i in range(0, 2)])
                )

        else:
            rates = total_results[params['bhat_infos'][0]]['rates']
            recs = total_results[params['bhat_infos'][0]]['recs']
        if rates[0][0] != 0.0001:
            rates = (
                [(0.0001, 1.0)]
                + [(i, 1.0) for i in np.arange(0.1, rates[0][0], 0.1)]
                + rates
            )
            recs = (
                [(0.0001, nrzero)]
                + [(i, nrzero) for i in np.arange(0.1, recs[0][0], 0.1)]
                + recs
            )
        if rates[-1][0] != 3.1:
            # Strange hack as np.arange(1.8, 2.0, 0.1) -> [1.8, 1.9] but np.arange(1.8, 2.1, 0.1) -> [1.8, 1.9, 2.0, 2.1] ....
            rates = rates + [(i, 0.0) for i in np.arange(rates[-1][0] + 0.1, 3.2, 0.1)]
            recs = recs + [(i, 0.0) for i in np.arange(recs[-1][0] + 0.1, 3.2, 0.1)]

        m = next(marker)
        d = next(dashed)
        color_tmp = next(colors)
        if d != 'dashed':
            ax_suc.plot(
                *zip(*rates), linestyle=d, color=color_tmp, label='{}'.format(nrzero)
            )
            ax_rec.plot(
                *zip(*recs), linestyle=d, color=color_tmp, label='{}'.format(nrzero)
            )
        else:
            d_spacing = next(dashed_spacing)
            ax_suc.plot(
                *zip(*rates),
                linestyle=d,
                dashes=d_spacing,
                color=color_tmp,
                label='{}'.format(nrzero)
            )
            ax_rec.plot(
                *zip(*recs),
                linestyle=d,
                dashes=d_spacing,
                color=color_tmp,
                label='{}'.format(nrzero)
            )

    x_axis_step = 4
    x_axis_ticks = list(np.arange(0.0, 3.1 + 0.2, 0.1))
    x_axis_label = [
        round(x_axis_ticks[i], 1) if i % x_axis_step == 0 else ''
        for i in range(len(x_axis_ticks))
    ]

    y_axis_step = 2
    y_axis_ticks = list(range(0, round(ax_rec.dataLim.y1) + 1, 32))
    y_axis_labels = [
        round(y_axis_ticks[i], 1) if i % y_axis_step == 0 else ''
        for i in range(len(y_axis_ticks))
    ]

    # Hack for adding '32':
    # y_axis_labels[1] = 32

    ax_rec.legend()
    ax_rec.set_xlim(0, 3.2)
    ax_rec.set_xticks(x_axis_ticks)
    ax_rec.set_xticklabels(x_axis_label)
    ax_rec.set_yticks(y_axis_ticks)
    ax_rec.set_yticklabels(y_axis_labels)

    ax_rec.set_xlabel("$\sigma$")
    ax_rec.set_ylabel("coefficients recovered")
    fig_rec.tight_layout(pad=0.05)
    fig_rec.savefig(params['result_file'] + '_rec.pdf')

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
    fig_suc.tight_layout(pad=0.05, rect=(-0.01, -0.03, 1, 1))
    fig_suc.savefig(params['result_file'] + '_suc.pdf')


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
        '-c',
        '--configfile',
        dest='configfile',
        metavar="filename",
        help="Config file name for plot configuration(*.json).",
        type=str,
        required=True,
        default=None,
    )
    parser.add_argument(
        '-nz',
        '--number-zeros',
        dest='nr_zeros',
        nargs="+",
        type=int,
        required=True,
        help="Specify the amount of zeros in resfile ([64 128 ...]",
    )
    args = parser.parse_args()

    if args.configfile is not None:
        with open(args.configfile, "r") as configfile:
            config = json.load(configfile)
            plt.rcParams.update(config)
    plot(args.filenames, args.nr_zeros)
