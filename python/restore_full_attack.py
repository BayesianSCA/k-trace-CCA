from full_attack import *

results_path = 'res_full_attack'

data = np.load(results_path + '.npz', allow_pickle=True)

results = data['results'][()]
print(results)

params = data['params'][()]
print(params)
