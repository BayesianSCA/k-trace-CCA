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
from multiprocessing import cpu_count

sys.path.append('lib/')

from helpers.experiment import Experiment
from helpers.params import get_argument_parser, get_params, print_params
from helpers.intt_builder import build_masked_graphs, build_unmasked_graph
from simulation.gen_input import INTTInputMasked, INTTInputUnmasked, BHatInfo, gen_skpv
from simulation.leak_data import LeakDataMasked, LeakDataUnmasked
from test import create_experiment_masked, create_experiment_unmasked
from simulation.kyber.reference.params import KYBER_N
from simulation.kyber.reference.reduce import barrett_reduce
from simulation.kyber.reference.poly import poly
from simulation.recover_sk_sparse import RecoverSk


def main():
    params = {
        'seed': 42,
        'runs': 1,
        'sigmas': [
            0.2,
        ],
        'attack_strategies': [1, 2],
        'unmasked': True,
        'total_steps': 100,
        'height': KYBER_N,
        'layers': 7,
        'kyber_k': 3,
        'abort_entropy': 0.1,
        'abort_entropy_diff': 0.5,
        'thread_count': cpu_count() - 1,
        'step_size': 10,
        'do_not_draw': False,
        'draw_labels': True,
        'no_bp': True,
        'result_file': 'res_full_attack',
    }

    results = {}
    for attack_strategy in params['attack_strategies']:
        attack_strategy = AttackStrategy.generate(attack_strategy)
        results[attack_strategy] = {}
        for sigma in params['sigmas']:
            results[attack_strategy][sigma] = {}
            results[attack_strategy][sigma]['nbr_success'] = 0
            print("Sigma: ", sigma)
            print("==============")
            # Terrible..
            params['sigma'] = sigma
            for i_run in range(params['runs']):
                results[attack_strategy][sigma][i_run] = {}
                seed = (
                    params['seed'] + i_run
                )  # This should be statistically independent with a decent PRNG
                results[attack_strategy][sigma][i_run]['seed'] = seed
                random.seed(seed)
                rng = np.random.default_rng(seed)

                try:
                    skpv_orig, skpv = gen_skpv(rng, params['height'], params['kyber_k'])
                    print("skpv (in reg domain): ", skpv_orig)
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    rec_sk = RecoverSk(height=params['height'], vec_k=params['kyber_k'])
                    for i_trace in range(attack_strategy.nbr_traces):
                        bhat_info = attack_strategy.bhat_info(i_trace)
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
                        if params['no_bp']:
                            if params['unmasked']:
                                inttinput = INTTInputUnmasked.generate(
                                    rng, params['height'], 3, skpv, bhat
                                )
                                c_hat = inttinput.get_intt_input()
                            else:
                                inttinput = INTTInputMasked.generate(
                                    rng, params['height'], 3, skpv, bhat
                                )
                                maskshare, skm = inttinput.get_intt_inputs()
                                c_hat = barrett_reduce(maskshare + skm)
                        else:
                            if params['unmasked']:
                                r = create_experiment_unmasked(params, rng, skpv, bhat)
                            else:
                                r = create_experiment_masked(params, rng, skpv, bhat)
                            r.run(params['total_steps'])
                            _, stats, _ = r.print_results(not params['unmasked'])
                            print("")
                            assert stats[
                                'success'
                            ], "Key part not successfully recovered"
                            c_hat = poly()
                            if params['unmasked']:
                                c_hat.coeffs = r.key
                            else:
                                key_len = len(r.key) // 2
                                maskshare = r.key[:key_len]
                                skm = r.key[key_len:]
                                c_hat.coeffs = list(
                                    map(
                                        lambda a, b: barrett_reduce(a + b),
                                        maskshare,
                                        skm,
                                    )
                                )

                        rec_sk.recover_sk_hat(bhat, c_hat)
                    print(">" * 80)
                    assert rec_sk.is_complete, "Key not completely recovered"
                    # print("skpv   (in reg domain): ", skpv_orig)
                    # print("rec_sk (in reg domain): ", rec_sk.sk)
                    assert (
                        skpv_orig == rec_sk.sk
                    ).all(), "Key not successfully recovered (in ntt domain)"
                    assert (
                        skpv == rec_sk.sk_hat_mont
                    ).all(), "Key not successfully recovered  (in reg domain)"
                except:
                    print("ERROR in run {} with sigma {}".format(i_run, sigma))
                    traceback.print_exc()
                    print("This is run will be counted as unsuccessful attack.")
                    results[attack_strategy][sigma][i_run]['success'] = False
                else:
                    results[attack_strategy][sigma][i_run]['success'] = True
                    results[attack_strategy][sigma]['nbr_success'] += 1
    print(">" * 80)
    print("Params :\n", params)
    print(">" * 80)
    print("Final Results :\n", results)

    np.savez_compressed(
        "{}".format(params['result_file']), results=results, params=params
    )


class AttackStrategy:
    def __init__(self, attack_strategy):
        self.attack_strategy = attack_strategy

    @staticmethod
    def generate(attack_strategy):
        return AttackStrategy(attack_strategy)

    def __hash__(self):
        return hash(self.attack_strategy)

    def __eq__(self, other):
        return (
            hasattr(other, 'attack_strategy')
            and self.attack_strategy == other.attack_strategy
        )

    def __repr__(self):
        return str(self.attack_strategy)

    @property
    def nbr_traces(self):
        if self.attack_strategy == 1:
            return 6
        elif self.attack_strategy == 2:
            return 12
        else:
            raise NotImplementedError("Attack strategy not implemented.")

    def bhat_info(self, i_trace):
        if self.attack_strategy == 1:
            blocks = [0, 1] if not i_trace & 1 else [2, 3]
            set_bhat = i_trace >> 1
            return BHatInfo.from_generated_bhat(blocks, set_bhat)
        elif self.attack_strategy == 2:
            blocks = [i_trace & 0b11]
            set_bhat = i_trace >> 2
            return BHatInfo.from_generated_bhat(blocks, set_bhat)
        else:
            raise NotImplementedError("Attack strategy not implemented.")


if __name__ == '__main__':
    main()
