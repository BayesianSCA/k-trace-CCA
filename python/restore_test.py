from test import *

results_path = 'results'

data = np.load(results_path + '.npz', allow_pickle=True)

total_results = data['total_results'][()]
print(total_results)

params = data['params'][()]
print(params)

for bhat_info in params['bhat_infos']:
    for sigma in params['sigmas']:
        for i_run in range(params['runs']):
            data = np.load(
                "{}_exp_{}_{}_{}.npz".format(
                    params['result_file'], bhat_info, sigma, i_run
                ),
                allow_pickle=True,
            )
            skpv = data['skpv'][()]
            bhat = data['bhat'][()]
            params_i = data['params'][()]
