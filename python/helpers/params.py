import argparse
import math
from multiprocessing import cpu_count
from simulation.gen_input import BHatInfo


def get_argument_parser():
    parser = argparse.ArgumentParser(
        description="""Simulate a HW leakage in a Kyber reference implemenation and run a belief propagaton on the leaked values.

    To run a full experiment, run e.g.: python -u python/test.py -r 5 -s 1.3 1.4 1.5 -n 64 --unmasked | tee results.txt
    """
    )

    parser.add_argument(
        '--runs',
        '-r',
        type=int,
        default=1,
        required=False,
        help='Specifies the number of graphs to create and run.',
    )
    parser.add_argument(
        '--sigmas',
        '-s',
        nargs='*',
        type=float,
        required=False,
        default=[0.2],
        help='Determines the variance of the error distribution of the hamming weight leakage.',
    )
    parser.add_argument(
        '--nbr-nonzeros',
        '-n',
        type=int,
        default=[64],
        nargs="*",
        help='Set the number of nonzero values, possible: 256, 192, 128, 64(, 32 - KYBER1024 only)',
    )
    parser.add_argument(
        '--type-nonzeros',
        type=str,
        default='default',
        help='Set the type of distribution of nonzero values of vector: default, rearranged',
    )
    parser.add_argument(
        '--kyber-k', '-k', type=int, default=3, help='Set Kyber-K, possible: 2, 3, 4'
    )
    parser.add_argument(
        '--unmasked', action='store_true', help='Creates an unmasked graph'
    )
    parser.add_argument(
        '--iterations',
        '-i',
        type=int,
        default=1000,
        required=False,
        help='Determines the number of iterations.',
    )
    parser.add_argument(
        '--threads',
        '-t',
        type=int,
        required=False,
        default=0,
        help='Number of threads to work with',
    )
    parser.add_argument(
        '--step-size',
        type=int,
        required=False,
        default=20,
        help='Determines the number of iterations after which stats are printed',
    )
    parser.add_argument(
        '--height',
        type=int,
        required=False,
        default=0,
        help='Determines the height of the INTT. If neither height or layers are given, layers=7 and height=256.',
    )
    parser.add_argument(
        '--layers',
        '-l',
        type=int,
        required=False,
        default=0,
        help='Determines the depth/layers of the INTT. If neither height or layers are given, layers=7 and height=256.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=False,
        default=362989,
        help='Passes a seed to the rng. If the seed 0, perfect value leakage is simulated.',
    )
    parser.add_argument(
        '--abort-entropy',
        type=float,
        required=False,
        default=0.1,
        help='Determines the the smallest entropy before aborting',
    )
    parser.add_argument(
        '--abort-entropy-diff',
        type=float,
        required=False,
        default=0.05,
        help='Determines the after what entropy change to abort.',
    )
    parser.add_argument(
        '--draw-labels',
        action='store_true',
        help='Determines whether node labels are drawn in the first step',
    )
    parser.add_argument('--draw', action='store_true', help='Disables drawing')
    parser.add_argument(
        '--check-validity',
        action='store_true',
        help='Enables a validity check in every iteration.',
    )
    parser.add_argument(
        '--print-entropy', action='store_true', help='Print entropy after every step.'
    )
    parser.add_argument(
        "--result-file",
        "-rf",
        type=str,
        help="File to save the final plot to",
        default="results",
    )
    parser.add_argument(
        '--abort-recovered',
        default=[200, 1],
        nargs=2,
        type=int,
        help='[number of steps] [minimum coefficients with rank 0] - aborts if after [number of steps] less than [minimum coefficients with rank 0] are of rank 0',
    )

    return parser


def print_params(args):
    print("")
    print("Parameters: {}".format(str(args).split('(')[1].split(')')[0]))
    print("")


def create_bhat_info(nbr_nonzeros, type_nonzeros, kyber_k):
    bhat_infos = []
    if type_nonzeros == 'default':
        for nbr_nz in nbr_nonzeros:
            if nbr_nz == 256:
                bhat_infos.append(BHatInfo.from_generated_bhat([(0, 1), (2, 3)]))
            elif nbr_nz == 192:
                bhat_infos.append(BHatInfo.from_generated_bhat([(0,), (1,), (2,)]))
            elif nbr_nz == 128:
                bhat_infos.append(BHatInfo.from_generated_bhat([(0,), (2,)]))
            elif nbr_nz == 64:
                bhat_infos.append(BHatInfo.from_generated_bhat([(0,)]))
            elif nbr_nz == 32:
                assert (
                    kyber_k == 4
                ), "32 nonzero coefficients only possible for KYBER-1024"
                bhat_infos.append(BHatInfo.from_generated_bhat_32([(0,)]))
            else:
                raise ValueError(
                    "Default is only for nbr non-zeros 32, 64, 128, 192, 256"
                )
    elif type_nonzeros == 'rearranged':
        for nbr_nz in nbr_nonzeros:
            if nbr_nz == 256:
                bhat_infos.append(BHatInfo.from_generated_bhat_rearrange(4))
            elif nbr_nz == 192:
                bhat_infos.append(BHatInfo.from_generated_bhat_rearrange(3))
            elif nbr_nz == 128:
                bhat_infos.append(BHatInfo.from_generated_bhat_rearrange(2))
            elif nbr_nz == 64:
                bhat_infos.append(BHatInfo.from_generated_bhat_rearrange(1))
            elif nbr_nz == 32:
                bhat_infos.append(BHatInfo.from_generated_bhat_rearrange(0.5))
            else:
                raise ValueError(
                    "Rearrange is only for nbr non-zeros 32, 64, 128, 192, 256"
                )
    # TODO: add random zeros again for small tests
    else:
        raise NotImplementedError(
            "Type-nonzeros : {} not implemented".format(type_nonzeros)
        )
    return bhat_infos


def get_params(args):

    thread_count = args.threads
    step_size = args.step_size
    total_steps = args.iterations
    height = args.height
    layers = args.layers
    seed = args.seed
    sigmas = args.sigmas
    abort_entropy = args.abort_entropy
    abort_entropy_diff = args.abort_entropy_diff
    draw_labels = args.draw_labels
    do_not_draw = not args.draw
    set_check_validity = args.check_validity
    kyber_k = args.kyber_k

    if seed == 0:
        seed = None

    if thread_count == 0:
        thread_count = cpu_count()

    if sigmas == [] or sigmas is None or sigmas[0] == 0:
        raise ValueError("Sigma cannot be 0")

    if layers == 0 or layers is None:
        if height is None or height == 0:
            height = 256
        layers = int(math.log2(height)) - 1

    if height == 0 or height is None:
        height = 2 ** (layers + 1)

    bhat_infos = create_bhat_info(args.nbr_nonzeros, args.type_nonzeros, kyber_k)

    return {
        'thread_count': thread_count,
        'step_size': step_size,
        'total_steps': total_steps,
        'height': height,
        'layers': layers,
        'seed': seed,
        'sigmas': sigmas,
        'abort_entropy': abort_entropy,
        'abort_entropy_diff': abort_entropy_diff,
        'draw_labels': draw_labels,
        'do_not_draw': do_not_draw,
        'set_check_validity': set_check_validity,
        'bhat_infos': bhat_infos,
        'runs': args.runs,
        'kyber_k': kyber_k,
        'print_entropy': args.print_entropy,
        'unmasked': args.unmasked,
        'result_file': args.result_file,
        'abort_recovered': args.abort_recovered,
    }
