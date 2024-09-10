from scan import reentrancy_call
import os
from slither.slither import Slither
import solidity
from dependence_graph.IVFG import IVFG
import json
from node2vec import Node2Vec
from edge2vec import Edge2Vec
import time
import math
import threading
from dependence_graph.VNode import VNode
from slither.slithir.operations import Call, SolidityCall, Binary, BinaryType, Unary, Assignment, InternalCall
from slither.slithir.operations.transfer import Transfer
from dependence_graph.CallGraph import CallGraph
from slither.core.cfg.node import NodeType
import sys

class extract_antireentrancy_graph:
    def __init__(self, root) -> None:
        self.n2vec = Node2Vec('word2vec.model')
        self.e2vec = Edge2Vec()
        self.save_root = root
        os.makedirs(self.save_root, exist_ok=True)
        self.graph_num = -1

    def is_modifier_call(self, node):
        if isinstance(node, str):
            return False
        for ir in node.irs:
            if isinstance(ir, InternalCall):
                if ir.is_modifier_call:
                    return True
        return False

    def filter_function_node(self, subgraph_nodes, external_data_deps, filter_functions: list):
        scope_nodes = CallGraph.extract_path_graph_nodes(filter_functions)
        filter_nodes = set()
        for n in subgraph_nodes:
            n: VNode
            if n.is_start_contract:
                filter_nodes.add(n)
            elif n.node in scope_nodes:
                filter_nodes.add(n)
        for e in external_data_deps:
            if e.head.node not in scope_nodes:
                continue
            filter_nodes.add(e.tail)
        return filter_nodes

    def extract_from_a_file(self, full_path, thread_lock=None):
        solc_version = solidity.get_solc(full_path)
        file_dir = full_path.split('/')[-1]
        if thread_lock:
            thread_lock.acquire()
        os.makedirs(os.path.join(self.save_root, file_dir), exist_ok=True)
        if thread_lock:
            thread_lock.release()
        self.graph_num = -1
        if solc_version is None:
            print(f'cannot get a correct solc version for {full_path}')
            return -1
        try:
            sl = Slither(full_path, solc=solc_version)
        except Exception as e:
            print(f'compilation for {full_path} failed')
            return -1
        scan_reentrancy = reentrancy_call(sl)
        external_calls = scan_reentrancy.extract()
        contract2cn = {}

        for cn, contained_function, vars_written in external_calls:
            temp = contract2cn.get(cn.function.contract, list())
            temp.append((cn, contained_function, vars_written))
            contract2cn[cn.function.contract] = temp

        for cont, call_list in contract2cn.items():
            self.n2vec: Node2Vec
            self.n2vec.set_context(cont)
            ivg = IVFG(cont)
            ivg.build_IVFG()
            for c, contained_function, vars_written in call_list:
                start = time.time()
                dst = IVFG.extract_destination(c)
                subgraph_nodes, external_data_deps = ivg.extract_subgraph_nodes(dst, c, 300)
                end = time.time()
                callgraph = CallGraph(sl)
                callgraph.build_call_graph()
                paths = callgraph.extract_func_call_list(c.function, contained_function)
                for path in paths:
                    self.graph_num += 1
                    graph_data = {'nodes': [], 'start_node': -1, 'external_node': -1, 'edges': [], 'edge_index': [],
                                  'file': full_path, 'call_expression': str(c), 'source_mapping': c.source_mapping_str,
                                  'modifier_calls': [], 'function_call_list': []}
                    modifiers_calls_dict = {}
                    function_call_list = [f.function.name for f in path]
                    graph_data['function_call_list'] = function_call_list

                    subgraph_nodes = self.filter_function_node(subgraph_nodes, external_data_deps, path)
                    subgraph_edges = ivg.extract_subgraph_edges(subgraph_nodes)
                    elen = len(subgraph_edges)
                    graph_data['edge_index'] = [[0 for _ in range(elen)], [0 for _ in range(elen)]]
                    node2id = {}
                    nid = 0
                    for ir in c.irs:
                        if isinstance(ir, Transfer):
                            graph_data['transfer'] = True
                        elif 'swapExact' in ir.function.name and \
                                'For' in ir.function.name \
                                and 'SupportingFee' in ir.function.name:
                            graph_data['swapExact'] = True
                        elif ir.function.name == 'addLiquidityETH':
                            graph_data['addLiquidity'] = True
                    for n in subgraph_nodes:
                        n: VNode
                        if n.node == c:
                            node_vec = self.n2vec.node2vec(n, is_external_node=True)
                        else:
                            node_vec = self.n2vec.node2vec(n, is_external_node=False)
                        assert node_vec is not None
                        graph_data['nodes'].append(list(node_vec))
                        node2id[n] = nid
                        if n.is_start_contract:
                            graph_data['start_node'] = nid
                        if n.node == c:
                            graph_data['external_node'] = nid
                        if not n.is_start_contract and n.node.type != NodeType.PLACEHOLDER \
                                and n.function in cont.modifiers:
                            modifiers_calls_dict[n.function.name] = modifiers_calls_dict.get(n.function.name, []) + [
                                nid]
                        nid += 1
                    graph_data['modifier_calls'] = list(modifiers_calls_dict.values())
                    eid = 0
                    for e in subgraph_edges:
                        tail = e.tail
                        head = e.head
                        source_id = node2id[tail]
                        target_id = node2id[head]
                        graph_data['edge_index'][0][eid] = source_id
                        graph_data['edge_index'][1][eid] = target_id
                        edge_vec = self.e2vec.edge2vec(e)
                        assert edge_vec is not None
                        graph_data['edges'].append(list(edge_vec))
                        eid += 1
                    if thread_lock:
                        thread_lock.acquire()
                    try:
                        with open(os.path.join(self.save_root, file_dir, f'graph_{self.graph_num}.json'), 'w') as f:
                            json.dump(graph_data, f)
                    except Exception as e:
                        print(e)
                    if thread_lock:
                        thread_lock.release()

                if thread_lock:
                    thread_lock.acquire()
                with open('record_exe_time.txt', 'a') as f:
                    f.write(f'{end-start}, {len(ivg.ivfg_nodes)}, {len(ivg.ivfg_data_edges) + len(ivg.ivfg_control_edges)}\n')
                print(f'\t[{end - start} secs Done] {full_path} extracted!')
                if thread_lock:
                    thread_lock.release()
        return self.graph_num

    def extract(self, thread_num=8):
        # i = -1
        file_list = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith('.sol'):
                    file_list.append(os.path.join(root, file))
                    # i += 1
        file_len = len(file_list)
        ind = 0
        while True:
            for k in range(thread_num):
                ind += k
                if ind >= file_len:
                    break
                file = file_list[ind]
                t = threading.Thread(target=self.extract_from_a_file, args=(file,))


class multithread_task():
    def __init__(self, analyze_path, save_root, thread_num=8):
        self.thread_lock = threading.Lock()
        self.file_list = []
        self.analyze_path = analyze_path
        self.root = save_root
        self.count = 0
        self.valid_count = 0
        self.graph_count = 0
        os.makedirs(save_root, exist_ok=True)
        for root, dirs, files in os.walk(self.analyze_path):
            for file in files:
                if file.endswith('.sol'):
                    self.file_list.append(os.path.join(root, file))
                    # i += 1
        self.thread_num = thread_num

    def thread_func(self, root, files, thread_lock):
        Re_graph = extract_antireentrancy_graph(root=root)
        for file in files:
            self.thread_lock.acquire()
            self.count += 1
            print(f'processing {self.count}-th file: {file}--------------')
            self.thread_lock.release()
            Re_graph.extract_from_a_file(file, thread_lock)
        return 0

    def divide(self, l):
        s = []
        offset = math.ceil(len(l) / self.thread_num)
        # print(offset)
        for i in range(0, len(l), offset):
            b = l[i: i + offset]
            s.append(b)
        return s

    def run(self):
        groups = self.divide(self.file_list)
        self.count = 0
        # print(groups)
        for files in groups:
            t = threading.Thread(target=self.thread_func, args=(self.root, files, self.thread_lock))
            t.start()


def test(save_root, analyze_path):
    os.makedirs(save_root, exist_ok=True)
    valid = 0
    graph_num = 0
    graph_dist = []
    count = 0
    Re_graph = extract_antireentrancy_graph(root=save_root)
    for root, dirs, files in os.walk(analyze_path):
        for file in files:
            if file.endswith('.sol'):
                count += 1
                file_path = os.path.join(root, file)
                print(f'------processing {count}-th file----------')
                gn = Re_graph.extract_from_a_file(file_path)
                if gn != -1:
                    valid += 1
                    graph_num += gn
                    graph_dist.append(gn)
                print(valid, graph_num)
    print('===========================')
    print(valid, graph_num, graph_dist)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('incorrect parameter number')
        sys.exit()
    source_raw_data = sys.argv[1]
    extracted_graph_data = sys.argv[2]
    multithread = int(sys.argv[3])
    mt = multithread_task(source_raw_data, extracted_graph_data, 6)
    mt.run()