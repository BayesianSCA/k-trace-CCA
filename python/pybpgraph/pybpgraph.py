import scipy.stats
import networkx as nx
import matplotlib.pyplot as plt
import math


class PyBPGraph:
    def __init__(self, GraphType, thread_count=1, *type_construtor_args):
        self.g = GraphType(*type_construtor_args)
        self.G = nx.Graph()
        self.results = None
        self.thread_count = thread_count

    def ini(self, *args):
        self.g.ini(*args)

    def set_check_validity(self, value):
        self.g.set_check_validity(value)

    def add_node(
        self,
        function_name,
        pos=None,
        name=None,
        color=(140, 140, 155),
        has_result=False,
        networx_node_args={},
        rust_node_args=[],
    ):

        if color is not None:
            networx_node_args['color'] = color
        if name is not None:
            networx_node_args['name'] = name
        else:
            name = "unnamed node"
        if has_result is not None:
            networx_node_args['has_result'] = has_result
        if pos is not None:
            networx_node_args['pos'] = pos

        f = getattr(self.g, function_name)
        idx = f(name, *rust_node_args)
        self.G.add_node(idx, **networx_node_args)
        return idx

    def add_variable_node(
        self,
        name,
        pos,
        vrange,
        prior=None,
        color=(65, 65, 230),
        group='inner',
        self_start=True,
    ):
        max_entropy = math.log2(vrange[1] + 1 - vrange[0])
        networx_node_args = {
            'group': group,
            'max_entropy': max_entropy,
            'v_range': vrange,
        }
        idx = self.add_node(
            'add_threaded_var_node',
            pos,
            name,
            color,
            True,
            networx_node_args=networx_node_args,
            rust_node_args=[prior, *vrange, self_start],
        )
        return idx

    def add_butterfly_node(self, name, pos, zeta, color=(140, 140, 155)):
        idx = self.add_node(
            'add_butterfly_node', name=name, pos=pos, color=color, rust_node_args=[zeta]
        )
        return idx

    def get_entropy_dict(self, group='inner'):
        nodes_attr = self.G.nodes(data=True)
        entropy = {}
        for node in self.G.nodes():
            if not nodes_attr[node].get('has_result', False):
                continue
            if nodes_attr[node].get('group', 'default') != group:
                continue
            entropy[node] = self.get_entropy(node)
        return entropy

    def set_thread_count(self, thread_count):
        self.thread_count = thread_count

    def compute_entropy_dict_parallel(self):
        nodes_attr = self.G.nodes(data=True)
        nodes_list = list(
            filter(lambda n: nodes_attr[n].get('has_result', False), self.G.nodes())
        )
        print("Computing results..")
        self.results = self.g.get_results(nodes_list, self.thread_count)
        print(f"Computed {len(self.results)} results.")

    def add_edge(self, fromnode, tonode):
        self.g.add_edge(fromnode, tonode)
        self.G.add_edge(fromnode, tonode)

    def get_result(self, nodeindex):
        if not self.results:
            self.compute_entropy_dict_parallel()
        res = self.results[nodeindex]
        if res:
            return res[0]
        return res
        # total = sum([res[val] for val in res])
        # if total != 0:
        #    return {val: res[val] / total for val in res}
        # return None

    def propagate(self, steps, threads):
        self.results = None
        self.g.propagate(steps, threads)

    def get_entropy(self, nodeindex):
        if not self.results:
            self.compute_entropy_dict_parallel()
        nodes_attr = self.G.nodes(data=True)
        if not nodes_attr[nodeindex].get('has_result', False):
            return None
        res = self.results[nodeindex]
        if res is None:
            res_ent = nodes_attr[nodeindex].get('max_entropy')
        else:
            res_ent = res[1]
            if res_ent > 13:
                print("{}".format({res[0][v] for v in res[0] if res[0][v] > 0.0001}))
                print(nodeindex)
                print(nodes_attr[nodeindex])
                assert False
        return res_ent

    def draw(self, name="", withlables=False, withpos=True, withcolor=True):

        plt.figure()
        plt.title(name)
        nodes_attr = self.G.nodes(data=True)
        if withpos:
            pos = nx.get_node_attributes(self.G, 'pos')
        else:
            pos = None
        if withlables:
            labels = nx.get_node_attributes(self.G, 'name')
        else:
            labels = None
        if withcolor:
            colors = [norm_rgb(nodes_attr[n]['color']) for n in self.G.nodes()]
        else:
            colors = None
        for n in self.G.nodes():
            ent = None
            if nodes_attr[n].get('has_result', False):
                ent = self.get_entropy(n)
                max_ent = nodes_attr[n].get('max_entropy', None)
                if ent and max_ent:
                    colors[n] = (*colors[n], 1.0 - max((0.9 * ent / max_ent), 0.1))

        nx.draw(self.G, pos, node_color=colors, with_labels=withlables, labels=labels)

    def show(self):
        plt.show()

    def len(self):
        return len(self.G.nodes)


def norm_rgb(colors):
    return (norm_color(colors[0]), norm_color(colors[1]), norm_color(colors[2]))


def norm_color(color):
    if color > 1:
        return color / 255
    return color
