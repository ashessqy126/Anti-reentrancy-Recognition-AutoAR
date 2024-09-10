import gensim
import numpy as np
from split_words import split_words
from slither.core.declarations import Contract
from slither.slithir.variables import Constant
from slither.core.cfg.node import Node, NodeType
import json
from slither.slither import Slither
import solidity
from slither.slithir.operations import (Call, SolidityCall, Binary, BinaryType, Unary, LibraryCall,
                                        Assignment, InternalCall, TypeConversion, HighLevelCall, LowLevelCall)
from slither.slithir.operations.transfer import Transfer
from slither.core.expressions import UnaryOperationType
from slither.core.solidity_types import ArrayType, ElementaryType, UserDefinedType, Type, MappingType
from slither.core.variables.local_variable import LocalVariable
from slither.core.variables.state_variable import StateVariable
from slither.core.declarations.solidity_variables import SolidityVariableComposed
from slither.slithir.variables.reference import ReferenceVariable
from dependence_graph.VNode import VNode
import string
import re

class Node2Vec:
    RESERVED = ["&&", "||", '!=', '==', 'msg.sender', 'tx.origin']
    UNKNOWN_WORD = '_unknown_word_'
    BASIC_DIM = 100

    #Variables
    LOCAL_VAR_READ = BASIC_DIM + 0
    LOCAL_VAR_WRITTEN = BASIC_DIM + 1
    STATE_VAR_READ = BASIC_DIM + 2
    STATE_VAR_WRITTEN = BASIC_DIM + 3
    STATE_ADDRESS = BASIC_DIM + 4 #ignore
    
    CONSTANT_VAR_NUMBER = BASIC_DIM + 10
    INT_NUMBER = BASIC_DIM + 11
    ARRAY_NUMBER = BASIC_DIM + 12
    BOOL_NUMBER = BASIC_DIM + 13
    BYTES32 = BASIC_DIM + 14
    CONTRACT_NUMBER = BASIC_DIM + 25
    SOLIDITY_VAR_NUMBER = BASIC_DIM + 26
    PRIVATE_VAR_NUMBER = BASIC_DIM + 27#
    
    TX_ORIGIN = BASIC_DIM + 19
    MSG_SENDER = BASIC_DIM + 20
    OWNER = BASIC_DIM + 21 #ignore
    BLOCK_TIMESTAMP = BASIC_DIM + 22
    MSG_VALUE = BASIC_DIM + 23
    ADDR_NUMBER = BASIC_DIM + 24
    

    #Function Calls
    INTERNAL_FUNC_CALL_NUMBER = BASIC_DIM + 5
    PUBLIC_FUNC_CALL_NUMBER = BASIC_DIM + 6
    HIGH_LEVEL_CALL_NUMBER = BASIC_DIM + 7
    LOW_LEVEL_CALL_NUMBER = BASIC_DIM + 8
    SEND_ETH = BASIC_DIM + 9
    INTERNAL_TRANSFER = BASIC_DIM + 62

    #Operator
    RELATION_OP = BASIC_DIM + 15
    LOGIC_OP = BASIC_DIM + 16
    ARITH_OP = BASIC_DIM + 17
    BIT_OP = BASIC_DIM + 18


    # Node type
    ENTRYPOINT_TYPE = BASIC_DIM + 28  # no expression
    EXPRESSION_TYPE = BASIC_DIM + 29  # normal case
    RETURN_TYPE = BASIC_DIM + 30  # RETURN may contain an expression
    IF_TYPE = BASIC_DIM + 31
    VARIABLE_TYPE = BASIC_DIM + 32 # Declaration of variable
    ASSEMBLY_TYPE = BASIC_DIM + 33
    IFLOOP_TYPE = BASIC_DIM + 34
    ENDIF_TYPE = BASIC_DIM + 35 # ENDIF node source mapping points to the if/else body
    STARTLOOP_TYPE = BASIC_DIM + 36  # STARTLOOP node source mapping points to the entire loop body
    ENDLOOP_TYPE = BASIC_DIM + 37 # ENDLOOP node source mapping points to the entire loop body
    THROW_TYPE = BASIC_DIM + 38
    BREAK_TYPE = BASIC_DIM + 39
    CONTINUE_TYPE = BASIC_DIM + 40
    PLACEHOLDER_TYPE = BASIC_DIM + 41
    TRY_TYPE = BASIC_DIM + 42
    CATCH_TYPE = BASIC_DIM + 43
    OTHER_ENTRYPOINT_TYPE = BASIC_DIM + 44
    CONTRACT_START_TYPE = BASIC_DIM + 45

    # Solidity call
    REQUIRE_CALL = BASIC_DIM + 46
    REVERT_CALL = BASIC_DIM + 47
    KECCAK_CALL = BASIC_DIM + 48
    SHA256_CALL = BASIC_DIM + 49
    RIPEMD_CALL = BASIC_DIM + 50
    ECRECOVER_CALL = BASIC_DIM + 51
    SELFDES_CALL = BASIC_DIM + 52
    SUICIDE_CALL = BASIC_DIM + 53
    BLOCKHASH_CALL = BASIC_DIM + 54
    BALANCE_CALL = BASIC_DIM + 55
    ABI_ENCODE_CALL = BASIC_DIM + 56
    ABI_ENCODE_PAC_CALL = BASIC_DIM + 57
    # TYPE_CALL = BASIC_DIM + 58 #ignore
    

    # Library
    LIB_CALL_NUMBER = BASIC_DIM + 58
    SENDER_VERIFY = BASIC_DIM + 59 #add one
    # MODIFIER_CALL_NUMBER = BASIC_DIM + 60
    #Contract
    REENTRANT_CONTRACT = BASIC_DIM + 60 #ignore


    Nodetype_mapping = {
        NodeType.ENTRYPOINT: ENTRYPOINT_TYPE,
        NodeType.VARIABLE: VARIABLE_TYPE,
        NodeType.OTHER_ENTRYPOINT: OTHER_ENTRYPOINT_TYPE,
        NodeType.THROW: THROW_TYPE,
        NodeType.IF: IF_TYPE,
        NodeType.ASSEMBLY: ASSEMBLY_TYPE,
        NodeType.BREAK: BREAK_TYPE,
        NodeType.CATCH: CATCH_TYPE,
        NodeType.CONTINUE: CONTINUE_TYPE,
        NodeType.ENDIF: ENDIF_TYPE,
        NodeType.ENDLOOP: ENDLOOP_TYPE,
        NodeType.EXPRESSION: EXPRESSION_TYPE,
        NodeType.IFLOOP: IFLOOP_TYPE,
        NodeType.PLACEHOLDER: PLACEHOLDER_TYPE,
        NodeType.RETURN: RETURN_TYPE,
        NodeType.STARTLOOP: STARTLOOP_TYPE,
        NodeType.TRY: TRY_TYPE
    }

    Soliditycall_mapping = {
        'require(': REQUIRE_CALL,
        'revert(': REVERT_CALL,
        'keccak256(': KECCAK_CALL,
        'sha256(': SHA256_CALL,
        'ripemd160(': RIPEMD_CALL,
        'ecrecover(': ECRECOVER_CALL,
        'selfdestruct(': SELFDES_CALL,
        'suicide(': SUICIDE_CALL,
        'blockhash(': BLOCKHASH_CALL,
        'balance(': BALANCE_CALL,
        'abi.encode(': ABI_ENCODE_CALL,
        'abi.encodePacked(': ABI_ENCODE_PAC_CALL
    }

    def __init__(self, model_path):
        self.wv_model = gensim.models.Word2Vec.load(model_path)
        self.dim = self.wv_model.wv.vectors.shape[1]
        self.context = None
        self.contract_sentences = None
        self.node2id = None
        self.sentences_tfidf = None
        self.extend_dim = self.dim + 60

    @staticmethod
    def convert_number(sentence: list):
        new_sentence = []
        for s in sentence:
            if s.isdigit():
                new_sentence.append('constant')
            elif '0x' in s or s[2:].isdigit():
                new_sentence.append('constant')
            else:
                new_sentence.append(s)
        return new_sentence

    @staticmethod
    def extract_constant_strs(node: Node):
        constants = []
        for ir in node.irs:
            for r in ir.read:
                if isinstance(r, Constant) and isinstance(r.value, str) and len(r.value) > 1:
                    constants.append(r)
        return constants

    @staticmethod
    def replace_var_name(txt, old_var_name, new_var_name):
        # words = split_words(txt, RESERVED, lower=False)
        punctuation = string.punctuation.replace('_', '') + " "
        start = 0
        str_len = len(txt)
        for i in range(str_len):
            if txt[start] not in punctuation:
                if txt[i] in punctuation:
                    end = i
                    var_name = txt[start:end]
                    if var_name == old_var_name:
                        return txt[0:start] + new_var_name + txt[end:]
                    start = end
            else:
                if txt[i] not in punctuation:
                    end = i
                    start = end
        return txt

    @staticmethod
    def replace_var_name(txt, old_var_name, new_var_name):
        # words = split_words(txt, RESERVED, lower=False)
        punctuation = string.punctuation.replace('_', '') + " "
        start = 0
        str_len = len(txt)
        for i in range(str_len):
            if txt[start] not in punctuation:
                if txt[i] in punctuation:
                    end = i
                    var_name = txt[start:end]
                    if var_name == old_var_name:
                        return txt[0:start] + new_var_name + txt[end:]
                    start = end
            else:
                if txt[i] not in punctuation:
                    end = i
                    start = end
        return txt

    @staticmethod
    def extract_external_address(node: Node):
        addrs = []
        convert_addrs = []
        RESERVED = ['msg.sender', 'tx.origin', 'Owner', 'owner', '_owner', '_Owner']
        for ir in node.irs:
            if isinstance(ir, TypeConversion) and isinstance(ir.type, ElementaryType) and ir.type.name == 'address':
                pattern = "address\s*\(\s*(\w*)\s*\)"
                matchObj = re.search(pattern, str(node))
                if matchObj:
                    if matchObj.group(1) in RESERVED:
                        continue
                    if matchObj.group() not in convert_addrs:
                        convert_addrs.append(matchObj.group())
            if isinstance(ir, TypeConversion) and isinstance(ir.type, UserDefinedType) and isinstance(ir.type.type,
                                                                                                      Contract):
                pattern = f"{ir.type}\s*\(\s*(\w*)\s*\)"
                matchObj = re.search(pattern, str(node))
                if matchObj:
                    if matchObj.group(1) in RESERVED:
                        continue
                    if matchObj.group() not in convert_addrs:
                        convert_addrs.append(matchObj.group())
        for c in node.variables_read:
            if str(c) in RESERVED:
                continue
            if isinstance(c, Type):
                continue
            if hasattr(c, 'type') and isinstance(c.type, ElementaryType) and c.type.name == 'address':
                if str(c) not in addrs:
                    addrs.append(str(c))
            elif hasattr(c, 'type') and isinstance(c.type, UserDefinedType) and isinstance(c.type.type, Contract):
                if str(c) not in addrs:
                    addrs.append(str(c))
        return convert_addrs, addrs

    def normalize(self, node: Node):
        expression = ""
        if node.expression:
            expression += " " + str(node.expression)
        elif node.variable_declaration:
            expression += " " + str(node.variable_declaration)
        txt = str(node.type) + expression
        for c in self.extract_constant_strs(node):
            txt = txt.replace(c.value, 'string_type')
        pattern = 'EXPRESSION\srequire\(\w+,\w+\)\((\w+\s*[<=>=&|]+\s*\w+),\s*\w*\)'
        m = re.match(pattern, txt)
        if m:
            txt = 'require ' + m.group(1)
        convert_addrs, addrs = self.extract_external_address(node)
        for addr in convert_addrs:
            txt = txt.replace(addr, 'addr_var')
        for addr in addrs:
            txt = self.replace_var_name(txt, addr, 'addr_var')
        return txt

    def set_context(self, cont):
        self.context = cont
        self.contract_sentences = []
        self.node2id = {}
        nid = 0
        for function in cont.functions_and_modifiers:
            for node in function.nodes:
                self.node2id[node] = nid
                nid += 1
                self.contract_sentences.append(self.code2sentence(node))
        self.node2id[f'{cont.name}.START_CONTRACT'] = nid
        self.contract_sentences.append(self.code2sentence(f'{cont.name}.START_CONTRACT'))
        self.sentences_tfidf = self.tfidf_model(self.contract_sentences)

    def tfidf_model(self, sentences):
        gensim_dictionary = gensim.corpora.Dictionary()
        gensim_corpus = [gensim_dictionary.doc2bow(sent, allow_update=True) for sent in sentences]
        tfidf = gensim.models.TfidfModel(gensim_corpus, smartirs='ntc')
        sentences_tfidf = []
        for sent in tfidf[gensim_corpus]:
            sentences_tfidf.append({gensim_dictionary[id]: np.around(frequency, decimals=2) for id, frequency in sent})
        return sentences_tfidf

    def code2vec(self, node):
        sentence = self.code2sentence(node)
        nid = self.node2id[node]
        tf = self.sentences_tfidf[nid]
        node_vec = np.zeros(self.dim)
        total_c = 0
        for w in set(sentence):
            c = tf[w]
            total_c += c
            if w in self.wv_model.wv.key_to_index:
                w_vec = self.wv_model.wv[w]
            else:
                w_vec = self.wv_model.wv[self.UNKNOWN_WORD]
            node_vec += c * w_vec
        return node_vec / total_c

    def code2sentence(self, node):
        if isinstance(node, str):
            sentence = node
        elif node.type == NodeType.OTHER_ENTRYPOINT:
            sentence = self.normalize(node)
            if node.variables_written:
                v = node.variables_written[0]
                sentence = sentence[0:16] + ' ' + str(v.type) + ' ' + sentence[16:]
        elif node.type == NodeType.ENTRYPOINT:
            sentence = f'ENTRY {node.function.signature}'
        elif node.type == NodeType.VARIABLE:
            sentence = self.normalize(node)
            sentence = sentence.replace(str(NodeType.VARIABLE), f'{node.variable_declaration.type}')
        else:
            sentence = self.normalize(node)
            sentence = sentence.replace('EXPRESSION', '')
        sentence = split_words(sentence, self.RESERVED)
        sentence = self.convert_number(sentence)
        return sentence

    def contain_external_call(self, node):
        for ir in node.irs:
            if isinstance(ir, HighLevelCall) or isinstance(ir, LowLevelCall):
                return ir
        return None

    def add_attr_feature(self, node: Node or str, vec, is_external_node=False):
        feature_weight = {}
        feature_weight_old = None
        with open('dimData.json') as f:
            feature_weight_old = json.load(f)
        for k, v in feature_weight_old.items():
            feature_weight[int(k)] = v

        feature_vec = np.zeros(self.extend_dim)
        feature_vec[:self.dim] = vec[:self.dim]
        if isinstance(node, str):
            feature_vec[self.CONTRACT_START_TYPE] = feature_weight[self.CONTRACT_START_TYPE]
            return feature_vec

        feature_vec[self.LOCAL_VAR_READ] = len(node.local_variables_read) * feature_weight[self.LOCAL_VAR_READ]
        feature_vec[self.LOCAL_VAR_WRITTEN] = len(node.local_variables_written) * feature_weight[self.LOCAL_VAR_WRITTEN]
        feature_vec[self.STATE_VAR_READ] = len(node.state_variables_read) * feature_weight[self.STATE_VAR_READ]
        feature_vec[self.STATE_VAR_WRITTEN] = len(node.state_variables_written) * feature_weight[self.STATE_VAR_WRITTEN]
        node_expr = str(node)
        if 'msg.sender' in node_expr:
            feature_vec[self.MSG_SENDER] += feature_weight[self.MSG_SENDER]
        if '_msgSender' in node_expr or '_msgsender' in node_expr or 'msgSender(' in node_expr:
            feature_vec[self.MSG_SENDER] += feature_weight[self.MSG_SENDER]
        if 'owner(' in node_expr:
            feature_vec[self.OWNER] += feature_weight[self.OWNER]
            # feature_vec[self.ADDR_NUMBER] += 1
        if 'tx.origin' in node_expr:
            feature_vec[self.TX_ORIGIN] += feature_weight[self.TX_ORIGIN]
        if 'require(' in node_expr and ('block.timestamp' in node_expr or 'block.number' in node_expr):
            feature_vec[self.BLOCK_TIMESTAMP] += feature_weight[self.BLOCK_TIMESTAMP]
        if 'require(' in node_expr and 'msg.value' in node_expr:
            feature_vec[self.MSG_VALUE] += feature_weight[self.MSG_VALUE]
        feature_vec[self.LIB_CALL_NUMBER] = len(node.library_calls) * feature_weight[self.LIB_CALL_NUMBER]
        # node type
        ntype_ind = self.Nodetype_mapping[node.type]
        feature_vec[ntype_ind] = feature_weight[ntype_ind]
        if node.type == NodeType.TRY:
            feature_vec[ntype_ind] = feature_weight[self.TRY_TYPE]
        if node.function in node.function.contract.modifiers and \
                ('Reentrant' in node.function.name or 'lock' in node.function.name):
            feature_vec[self.REENTRANT_CONTRACT] += feature_weight[self.REENTRANT_CONTRACT]
        # solidity call
        for ir in node.irs:
            if isinstance(ir, Transfer):
                feature_vec[self.INTERNAL_TRANSFER] += feature_weight[self.INTERNAL_TRANSFER]
            if isinstance(ir, InternalCall):
                # if ir.is_modifier_call:
                #     feature_vec[self.MODIFIER_CALL_NUMBER] += 1
                if ir.function.visibility in ['external', 'public']:
                    feature_vec[self.PUBLIC_FUNC_CALL_NUMBER] += feature_weight[self.PUBLIC_FUNC_CALL_NUMBER]
                else:
                    feature_vec[self.INTERNAL_FUNC_CALL_NUMBER] += feature_weight[self.INTERNAL_FUNC_CALL_NUMBER]
            if isinstance(ir, LibraryCall):
                if ir.function.name == 'recover' and str(ir.destination) == 'ECDSA' or \
                        ir.function.name == 'verify' and str(ir.destination) == 'MerkleProof':
                    feature_vec[self.SENDER_VERIFY] += feature_weight[self.SENDER_VERIFY]
            if isinstance(ir, HighLevelCall) and ir.function is not None and ir.can_reenter():
                feature_vec[self.HIGH_LEVEL_CALL_NUMBER] += feature_weight[self.HIGH_LEVEL_CALL_NUMBER]
            if isinstance(ir, LowLevelCall):
                feature_vec[self.LOW_LEVEL_CALL_NUMBER] += feature_weight[self.LOW_LEVEL_CALL_NUMBER]
            if hasattr(ir, "call_value") and ir.call_value is not None:
                feature_vec[self.SEND_ETH] += feature_weight[self.SEND_ETH]
            if isinstance(ir, Assignment):
                feature_vec[self.ARITH_OP] += feature_weight[self.ARITH_OP]
            if isinstance(ir, Binary):
                if ir.type in [BinaryType.ADDITION, BinaryType.SUBTRACTION, BinaryType.MULTIPLICATION,
                               BinaryType.DIVISION, BinaryType.POWER, BinaryType.MODULO]:
                    feature_vec[self.ARITH_OP] += feature_weight[self.ARITH_OP]
                if ir.type in [BinaryType.LESS, BinaryType.GREATER, BinaryType.LESS_EQUAL, BinaryType.GREATER_EQUAL,
                               BinaryType.EQUAL, BinaryType.NOT_EQUAL, BinaryType.ANDAND, BinaryType.OROR]:
                    feature_vec[self.RELATION_OP] += feature_weight[self.RELATION_OP]
                if ir.type in [BinaryType.ANDAND, BinaryType.OROR]:
                    feature_vec[self.LOGIC_OP] += feature_weight[self.LOGIC_OP]
                if ir.type in [BinaryType.LEFT_SHIFT, BinaryType.RIGHT_SHIFT, BinaryType.AND, BinaryType.CARET,
                               BinaryType.OR]:
                    feature_vec[self.BIT_OP] += feature_weight[self.BIT_OP]
            if isinstance(ir, Unary):
                if ir.type == UnaryOperationType.BANG:
                    feature_vec[self.LOGIC_OP] += feature_weight[self.LOGIC_OP]
                if ir.type == UnaryOperationType.TILD:
                    feature_vec[self.BIT_OP] += feature_weight[self.BIT_OP]
            if isinstance(ir, SolidityCall):
                if 'require(' in str(ir.expression):
                    feature_vec[self.Nodetype_mapping[NodeType.IF]] += feature_weight[self.IF_TYPE]
                elif 'keccak256(' in str(ir.expression):
                    feature_vec[self.KECCAK_CALL] += feature_weight[self.KECCAK_CALL]
                elif 'abi.encodePacked(' in str(ir.expression):
                    feature_vec[self.ABI_ENCODE_PAC_CALL] += feature_weight[self.ABI_ENCODE_PAC_CALL]
                else:
                    for key in self.Soliditycall_mapping.keys():
                        if key in str(ir.expression):
                            nsol_call_ind = self.Soliditycall_mapping[key]
                            feature_vec[nsol_call_ind] += feature_weight[nsol_call_ind]
                            break
            for r in ir.read:
                if isinstance(r, StateVariable) and hasattr(r, 'type'):
                    if isinstance(r.type, ElementaryType) and r.type.name == 'address' \
                            or (isinstance(r.type, UserDefinedType) and isinstance(r.type.type, Contract)):
                        feature_vec[self.STATE_ADDRESS] += feature_weight[self.STATE_ADDRESS]
                if str(r) in ['_owner', 'owner', 'Owner']:
                    feature_vec[self.OWNER] += 2
                    # feature_vec[self.ADDR_NUMBER] += 1
                if isinstance(r, StateVariable) and r.is_constant and hasattr(r, 'type') and 'int' in str(r.type):
                    feature_vec[self.CONSTANT_VAR_NUMBER] += feature_weight[self.CONTRACT_NUMBER]
                if isinstance(r, Constant):
                    if isinstance(r.value, bool):
                        feature_vec[self.BOOL_NUMBER] += feature_weight[self.BOOL_NUMBER]
                if hasattr(r, 'type') and isinstance(r.type, ArrayType):
                    feature_vec[self.ARRAY_NUMBER] += feature_weight[self.ARRAY_NUMBER]
                    if hasattr(r.type.type, 'name'):
                        if r.type.type.name == 'address':
                            feature_vec[self.ADDR_NUMBER] += feature_weight[self.ADDR_NUMBER]
                        if r.type.type.name == 'bytes32':
                            feature_vec[self.BYTES32] += feature_weight[self.BYTES32]
                if hasattr(r, 'type') and isinstance(r.type, ElementaryType) and r.type.name == 'bytes32':
                    feature_vec[self.BYTES32] += feature_weight[self.BYTES32]
                if hasattr(r, 'type') and isinstance(r.type, ElementaryType) and r.type.name == 'address':
                    feature_vec[self.ADDR_NUMBER] += feature_weight[self.ADDR_NUMBER]
                if hasattr(r, 'type') and isinstance(r.type, UserDefinedType) and isinstance(r.type.type, Contract):
                    feature_vec[self.CONTRACT_NUMBER] += feature_weight[self.CONTRACT_NUMBER]
                if isinstance(r, SolidityVariableComposed):
                    feature_vec[self.SOLIDITY_VAR_NUMBER] += feature_weight[self.SOLIDITY_VAR_NUMBER]
                if hasattr(r, 'type') and isinstance(r.type, ElementaryType) and 'int' in r.type.name:
                    feature_vec[self.INT_NUMBER] += feature_weight[self.INT_NUMBER]
                if hasattr(r, 'type') and isinstance(r.type, ElementaryType) and 'bool' in r.type.name:
                    feature_vec[self.BOOL_NUMBER] += feature_weight[self.BOOL_NUMBER]
                if hasattr(r, 'visibility') and r.visibility == 'private':
                    feature_vec[self.PRIVATE_VAR_NUMBER] += feature_weight[self.PRIVATE_VAR_NUMBER]
        return feature_vec

    def node2vec(self, n: VNode or Node, is_external_node=False):
        node = None
        if isinstance(n, VNode):
            node = n.node
        elif isinstance(n, Node):
            node = n
        if node is None:
            return None
        else:
            vec = self.code2vec(node)
            vec = self.add_attr_feature(node, vec, is_external_node)
            return vec


if __name__ == '__main__':
    n2vec = Node2Vec("word2vec.model")
    solc_compiler = solidity.get_solc('test.sol')
    sl = Slither('test.sol', solc=solc_compiler)
    for cont in sl.contracts:
        n2vec.set_context(cont)
        for function in cont.functions_and_modifiers:
            print(f'------------{function}------------')
            for node in function.nodes:
                vec = n2vec.code2vec(node)
                vec = n2vec.add_attr_feature(node, vec)
                print(node)
                print(f'\tnode vec:{vec}')