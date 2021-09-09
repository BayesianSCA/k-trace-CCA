# This is a rather large and complex class,
# but tries to capture all aspects of an experiment
class Experiment:
    def __init__(
        self,
        g,
        key,
        idx_key,
        len_zero_indices,
        min_entropy,
        max_entropy_diff,
        thread_count,
        step_size,
        do_not_draw,
        draw_labels,
        abort_recovered,
        is_masked,
        count_poss_values,
    ):
        self.g = g
        self.g.set_thread_count(thread_count)
        self.idx_key = idx_key
        self.key = key
        self.len_zero_indices = len_zero_indices
        self.min_entropy_cond = min_entropy
        self.max_entropy_diff_cond = max_entropy_diff
        self.step_size = step_size
        self.thread_count = thread_count
        self.step = 0
        # Inner nodes entropy
        self.last_entropy = self.g.get_entropy_dict()
        self.min_entropy = -1
        self.max_entropy = -1
        self.max_entropy_diff = -1
        self.avg_entropy = -1
        # Outer nodes entropy
        self.last_entropy_outer = self.g.get_entropy_dict(group='outer')
        self.min_entropy_outer = -1
        self.max_entropy_outer = -1
        self.max_entropy_diff_outer = -1
        self.avg_entropy_outer = -1
        # w nodes entropy
        self.last_entropy_w = self.g.get_entropy_dict(group='w')
        self.min_entropy_w = -1
        self.max_entropy_w = -1
        self.max_entropy_diff_w = -1
        self.avg_entropy_w = -1
        #
        self.do_not_draw = do_not_draw
        self.count_poss_values = count_poss_values
        self.draw_labels = draw_labels
        self.abort_recovered = abort_recovered
        self.abort_recovered_values = [0, len_zero_indices]
        self.is_masked = is_masked

    def calc_entropy(self):
        ent_dict = self.g.get_entropy_dict()
        ent_list = ent_dict.values()
        self.avg_entropy = sum(ent_list) / len(ent_list)
        self.min_entropy = min(ent_list)
        self.max_entropy = max(ent_list)
        diffs = [self.last_entropy[node] - ent_dict[node] for node in ent_dict]
        self.max_entropy_diff = max(diffs)
        self.last_entropy = ent_dict

    def calc_entropy_outer(self):
        ent_dict = self.g.get_entropy_dict(group='outer')
        ent_list = ent_dict.values()
        self.avg_entropy_outer = sum(ent_list) / len(ent_list)
        self.min_entropy_outer = min(ent_list)
        self.max_entropy_outer = max(ent_list)
        diffs = [self.last_entropy_outer[node] - ent_dict[node] for node in ent_dict]
        self.max_entropy_diff_outer = max(diffs)
        self.last_entropy_outer = ent_dict

    def calc_entropy_w(self):
        ent_dict = self.g.get_entropy_dict(group='w')
        ent_list = ent_dict.values()
        self.avg_entropy_w = sum(ent_list) / len(ent_list)
        self.min_entropy_w = min(ent_list)
        self.max_entropy_w = max(ent_list)
        diffs = [self.last_entropy_w[node] - ent_dict[node] for node in ent_dict]
        self.max_entropy_diff_w = max(diffs)
        self.last_entropy_w = ent_dict

    def check_entropy_conditions(self):
        """Abort if true"""
        if self.step == 0:
            return False
        if self.max_entropy <= self.min_entropy_cond:
            print("Minimal entropy reached. Aborting")
            return True
        if self.max_entropy_diff <= self.max_entropy_diff_cond:
            print("Minimal entropy difference reached. Aborting.")
            return True
        return False

    def check_abort_recovered(self):
        results, stats, unrecovered = self.get_result_stats()
        if stats['success']:
            print("All key-coefficients with rank 0. Successful Abort.")
            return True
        # only check for improvement every 'abort_recovered[0]' steps
        if self.step - self.abort_recovered_values[0] < self.abort_recovered[0]:
            return False
        self.print_results(results, stats, unrecovered)
        if (
            stats['rank_zero'] - self.abort_recovered_values[1]
            < self.abort_recovered[1]
        ):
            print(
                "{} with rank zero. Last check: {} with rank zero. Aborting.".format(
                    stats['rank_zero'], self.abort_recovered_values[1]
                )
            )
            return True
        print(
            "{} with rank zero. Last check: {} with rank zero. Continuing.".format(
                stats['rank_zero'], self.abort_recovered_values[1]
            )
        )
        self.abort_recovered_values[0] = self.step
        self.abort_recovered_values[1] = stats['rank_zero']

    def print_state(self):
        print("Propagated {} steps.".format(self.step))
        print("-" * 10)
        self.print_inner_state()
        # self.calc_entropy_outer()
        # self.print_outer_state()
        # self.calc_entropy_w()
        # self.print_w_state()
        print("-" * 10)

    def print_entropy_state(
        self, name, min_entropy, max_entropy, avg_entropy, max_entropy_diff
    ):
        print(f"{name}:")
        print(
            "Entropy between {:.4f} and {:.4f}. Average entropy at {:.8f}.".format(
                min_entropy, max_entropy, avg_entropy
            )
        )
        print("Maximal entropy change was {:.4f}.".format(max_entropy_diff))

    def print_w_state(self):
        self.print_entropy_state(
            "w-nodes",
            self.min_entropy_w,
            self.max_entropy_w,
            self.avg_entropy_w,
            self.max_entropy_diff_w,
        )

    def print_outer_state(self):
        self.print_entropy_state(
            "Outer nodes",
            self.min_entropy_outer,
            self.max_entropy_outer,
            self.avg_entropy_outer,
            self.max_entropy_diff_outer,
        )

    def print_inner_state(self):
        self.print_entropy_state(
            "Inner nodes",
            self.min_entropy,
            self.max_entropy,
            self.avg_entropy,
            self.max_entropy_diff,
        )

    def check_abort(self, max_step):
        if max_step != None and self.step >= max_step:
            print("Maximal number of iterations reached. Aborting.")
            return True
        return self.check_entropy_conditions() or self.check_abort_recovered()

    def run(self, max_step=None):
        print("Propagating with {} threads..".format(self.thread_count))
        while not self.check_abort(max_step):
            self.g.propagate(self.step_size, self.thread_count)
            self.calc_entropy()
            self.step += self.step_size
            self.print_state()
            if not self.do_not_draw:
                self.draw()

    def draw(self):
        self.g.draw(
            name="Step {} with average entropy at {}.".format(
                self.step, self.avg_entropy
            ),
            withlables=self.draw_labels,
        )
        self.g.show()

    def get_result_rank(self, idx, actual_value):
        result = self.g.get_result(idx)
        if result is None:
            v_range = self.g.G.nodes(data=True)[idx].get('v_range')
            result = v_range[1] - v_range[0] + 1
            return result

        result_list = sorted(result.items(), key=lambda vp: vp[1], reverse=True)
        index = -1
        for i, (v, _) in enumerate(result_list):
            if v == actual_value:
                index = i
                break
        rank = index
        return rank

    def get_results(self):
        results = []
        # Terrible code
        if self.is_masked:
            assert len(self.key) % 2 == 0
            key_len = len(self.key) // 2
            for actual_value0, idx0, actual_value1, idx1 in zip(
                self.key[:key_len],
                self.idx_key[:key_len],
                self.key[key_len:],
                self.idx_key[key_len:],
            ):
                rankm, rankskm = self.get_result_rank(
                    idx0, actual_value0
                ), self.get_result_rank(idx1, actual_value1)

                results.append(
                    {
                        'recovered': rankm == 0 and rankskm == 0,
                        'rank': (rankm + rankskm) / 2,
                        'index': idx0,
                    }
                )
        else:
            for actual_value, idx in zip(self.key, self.idx_key):
                rank = self.get_result_rank(idx, actual_value)
                results.append({'recovered': rank == 0, 'rank': rank, 'index': idx})
        return results

    def get_result_stats(self):
        results = self.get_results()
        rank0 = sum([1 for r in results if r['recovered']])
        avg_rank = sum(r['rank'] for r in results) / len(results)
        unrecovered = [r['index'] for r in results if not r['recovered']]
        return (
            results,
            {
                'rank_zero': rank0,
                'avg_rank': avg_rank,
                'success': rank0 == len(results),
            },
            unrecovered,
        )

    def print_results(self, results=None, stats=None, unrecovered=None):
        if results is None or stats is None or unrecovered is None:
            results, stats, unrecovered = self.get_result_stats()
        print("----------------------")
        print(
            "{} of {} coefficients of rank 0 ({} of each polynomial set to 0)".format(
                stats['rank_zero'], len(results), self.len_zero_indices
            )
        )
        print("Average rank is {}.".format(stats['avg_rank']))
        print("Indices of nodes with rank > 0: ", unrecovered)
        print("Success: {}".format(stats['success']))
        print("----------------------")
        return results, stats, unrecovered

    @staticmethod
    def from_params(
        g, key, idx_key, params, len_zero_indices, count_poss_values=3329 * 2
    ):
        assert len(key) == len(idx_key)
        return Experiment(
            g,
            key,
            idx_key,
            len_zero_indices,
            params['abort_entropy'],
            params['abort_entropy_diff'],
            params['thread_count'],
            params['step_size'],
            params['do_not_draw'],
            params['draw_labels'],
            params['abort_recovered'],
            not params['unmasked'],
            count_poss_values,
        )
