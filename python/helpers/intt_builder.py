from pybpgraph.pybpgraph import PyBPGraph
from ntt_bp import KyberRefINTTGraph
from simulation.kyber.reference.ntt import zetas
from simulation.gen_input import INTTInputMasked, INTTInputUnmasked
from simulation.leak_data import LeakDataMasked, LeakDataUnmasked


def build_kyber_intt_graph(
    leak_dict, height=256, layers=7, intermediate_values_range=[], g=None, shift_down=0
):
    if len(intermediate_values_range) == 0:
        intermediate_values_range = {
            (h, l): (-3328, 3328) for h in range(height) for l in range(layers)
        }
    if g == None:
        g = PyBPGraph(KyberRefINTTGraph)
    graph_start = g.len()
    for l in range(layers + 1):
        for h in range(0, height, 2):
            # if l == 0:
            vrange = intermediate_values_range[h, l]
            g.add_variable_node(
                name="v({},{})".format(l, h),
                pos=(l * 2, height - h - shift_down),
                prior=leak_dict[(h, l)],
                vrange=vrange,
            )
            vrange = intermediate_values_range[h + 1, l]
            g.add_variable_node(
                name="v({},{})".format(l, h + 1),
                pos=(l * 2, height - h - 1 - shift_down),
                prior=leak_dict[(h + 1, l)],
                vrange=vrange,
            )
    dist = 2
    k = height // 2 - 1
    layer = 0
    while dist <= height // 2:
        start = 0
        while start < height:
            # print("start: ", start)
            zeta = zetas[k]
            k -= 1
            j = start
            while j < start + dist:
                bn = g.add_butterfly_node(
                    name="bf({},{}, {})".format(layer, j, zeta),
                    pos=(layer * 2 + 1, height - j - shift_down),
                    zeta=zeta,
                )
                g.add_edge(layer * height + j + graph_start, bn)
                g.add_edge(layer * height + j + dist + graph_start, bn)
                g.add_edge((layer + 1) * height + j + graph_start, bn)
                g.add_edge((layer + 1) * height + j + dist + graph_start, bn)
                #
                j += 1

            start = j + dist
        dist <<= 1
        layer += 1
        if layer >= layers:
            break
    return g, [i + graph_start for i in range(0, height)]


def build_masked_graphs(leakdata, set_check_validity=False):
    print("Building graph for masked share..")
    g, mask_idx = build_kyber_intt_graph(
        leakdata.get_leak_dict_mask(),
        leakdata.height,
        leakdata.layers,
        leakdata.intermediate_values_range,
    )
    print("Building graph for skm share..")
    g, skm_idx = build_kyber_intt_graph(
        leakdata.get_leak_dict_skm(),
        leakdata.height,
        leakdata.layers,
        leakdata.intermediate_values_range,
        g,
        leakdata.height * 1.1,
    )
    g.set_check_validity(set_check_validity)
    g.ini()

    return g, mask_idx, skm_idx


def build_unmasked_graph(leakdata, set_check_validity=False):
    g, idx_keys = build_kyber_intt_graph(
        leakdata.get_leak_dict(),
        height=leakdata.height,
        layers=leakdata.layers,
        intermediate_values_range=leakdata.intermediate_values_range,
        shift_down=0,
    )
    g.set_check_validity(set_check_validity)
    g.ini()

    return g, idx_keys
