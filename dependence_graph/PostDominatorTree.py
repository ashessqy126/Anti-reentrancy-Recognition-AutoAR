from dependence_graph.VNode import VNode
from typing import Set, Dict, Union
from queue import Queue
from copy import copy
from slither.slither import Slither
from dependence_graph.cfg import CFG

class TreeNode():
    def __init__(self, _node) -> None:
        self._node = _node
        self._sons = []
        self._father = None

    def set_node(self, _node):
        self._node = _node

    def set_sons(self, _sons):
        self._sons = _sons

    def add_son(self, node):
        self._sons.append(node)

    def set_father(self, node):
        self._father = node

    @property
    def sons(self):
        return self._sons

    @property
    def node(self):
        return self._node

    @property
    def father(self):
        return self._father

    def __str__(self) -> str:
        return str(self._node)

class postDominatorTree():
    def __init__(self, all_nodes: Set[VNode], exit_node: VNode) -> None:
        self.dom_mapping = {}
        self.all_nodes = all_nodes
        self.exit_node = exit_node
        self.func_tree_nodes = {}
        self.func_entry_node = {}
        self.all_tree_nodes = []
        self.root = None
        self.cNode2Tnode = {}
        self._mapping()

    def _mapping(self):
        for cnode in self.all_nodes:
            func = cnode.function
            tnode = TreeNode(cnode)
            self.cNode2Tnode[cnode] = tnode
            self.func_tree_nodes[func] = self.func_tree_nodes.get(func, set()) | {tnode}
            if cnode.is_entry:
                self.func_entry_node[func] = cnode
            # if cnode.is_start:
            #     self.entry_node = cnode

    def intersect_immediate_post(self, n: VNode):
        temp = set()
        i = 0
        for son in n.sons:
            if i == 0:
                temp = copy(self.dom_mapping.get(son, set()))
                i = 1
            else:
                temp = temp & self.dom_mapping.get(son, set())
        return temp

    def compute_post_dominator(self):
        self.dom_mapping[self.exit_node] = {self.exit_node}
        for n in self.all_nodes - {self.exit_node}:
            self.dom_mapping[n] = self.all_nodes
        changed = True
        while changed:
            changed = False
            for n in self.all_nodes - {self.exit_node}:
                new_set = self.intersect_immediate_post(n).union({n})
                # print('>>>update', self.dom_mapping[n])
                if new_set != self.dom_mapping[n]:
                    # print('>>>>>>', original_dom, self.dom_mapping[n])
                    # if n.function is not None and n.function.name == 'tokenURI':
                    #     print('hahah')
                    self.dom_mapping[n] = new_set
                    changed = True

    def build_pdom_tree(self):
        Q = Queue()
        self.compute_post_dominator()
        # print(self.dom_mapping)
        # print(self.exit_node)
        n0 = self.cNode2Tnode[self.exit_node]
        # self.cNode2Tnode[self.exit_node] = n0
        self.all_tree_nodes.append(n0)
        Q.put(n0)
        # n0.set_sons(set())
        self.root = n0

        for n in self.all_nodes:
            # if n.function is not None and n.function.name == 'tokenURI':
            #     print('hahah')
            self.dom_mapping[n] = self.dom_mapping[n] - {n}

        while not Q.empty():
            m = Q.get()
            # if m.node.function and m.node.function.name == 'tokenURI':
            #     print('hhahah')
            for n in self.all_nodes:
                if not self.dom_mapping[n]:
                    continue
                if m.node in self.dom_mapping[n]:
                    self.dom_mapping[n] = self.dom_mapping[n] - {m.node}
                    if not self.dom_mapping[n]:
                        # print('m add son:', n)
                        child = self.cNode2Tnode[n]
                        # print('\t-', m.sons)
                        child.set_father(m)
                        m.add_son(child)
                        self.all_tree_nodes.append(child)
                        Q.put(child)

    def _preOrderVist(self, root: TreeNode, result: set):
        if root is None:
            return
        result.add(root)
        for son in root.sons:
            self._preOrderVist(son, result)

    def preOrderVist(self):
        result = set()
        self._preOrderVist(self.root, result)
        return result

    def print(self):
        for function, tnodes in self.func_tree_nodes.items():
            print(f'------------{function}------------')
            for tnode in tnodes:
                print(tnode.node, '-> sons:')
                if not tnode.sons:
                    print('\tNULL')
                for tson in tnode.sons:
                    print('\t-', tson.node)

    def LCA(self, a: TreeNode, b: TreeNode):
        visited = set()
        curr = b
        while curr:
            visited.add(curr)
            curr = curr.father
        curr = a
        while curr:
            if curr in visited:
                return curr
            curr = curr.father
        return curr

if __name__ == '__main__':
    sl = Slither('../test.sol')
    contract = sl.get_contract_from_name('GoblinRareApepeYC')[0]
    c = CFG(contract)
    c.build_cfg_graph()
    c.augment_cfg()
    # c.print_all_nodes()
    pDomTree = postDominatorTree(c.all_cfg_nodes, c.contract_end)
    pDomTree.build_pdom_tree()
    pDomTree.print()