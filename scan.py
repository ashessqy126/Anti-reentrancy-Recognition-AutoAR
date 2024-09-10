from detectors.reentrancy_eth import ReentrancyEth
import os
import solidity
from slither.slither import Slither
import logging
from typing import List
from collections import namedtuple
from slither.core.cfg.node import Node
import solidity

FindingKey = namedtuple("FindingKey", ["function", "calls", "send_eth"])
FindingValue = namedtuple("FindingValue", ["variable", "node", "nodes"])

class reentrancy_call:
    def __init__(self, sl: Slither):
        self.sl = sl

    def extract(self):
        logger_detector = logging.getLogger("Detectors")
        extracted_calls = set()
        for compilation_unit in self.sl.compilation_units:
            instance = ReentrancyEth(compilation_unit, self.sl, logger_detector)
            reentracies = instance.detect()
            varsWritten: List[FindingValue]
            tmp = set()
            for (function, calls, _), varsWritten in reentracies:
                calls = sorted(list(set(calls)), key=lambda x: x[0].node_id)
                vars_written = set()
                for v in varsWritten:
                    for w in v.nodes:
                        for st in w.state_variables_written:
                            vars_written.add(str(st))
                vars_written = '-'.join(vars_written)
                for (call_info, calls_list) in calls:
                    for c in calls_list:
                        # # if c == call_info:
                        if c.function not in tmp:
                            extracted_calls.add((c, function, vars_written))
                            tmp.add(c.function)
                        break
                        # break
                    # break
        return extracted_calls


if __name__ == '__main__':
    solc_compiler = solidity.get_solc('etherscan/0x0a0b44bb51f857b0ca8aded28fdb90414afe9d40/0x0a0b44bb51f857b0ca8aded28fdb90414afe9d40.sol')
    sl = Slither('etherscan/0x0a0b44bb51f857b0ca8aded28fdb90414afe9d40/0x0a0b44bb51f857b0ca8aded28fdb90414afe9d40.sol',
                 solc=solc_compiler)
    scan_reentrancy = reentrancy_call(sl)
    dangerous_calls = scan_reentrancy.extract()
    # print(dangerous_calls)
    print('dangerous call:')
    for c in dangerous_calls:
        print('\t', c)
    # i = 0
    # for root, dirs, files in os.walk('etherscan/'):
    #     for file in files:
    #         if file.endswith('.sol'):
    #             i += 1
    #             print(f'scanning the {i}-th file: {file}-----------------')
    #             full_path = os.path.join(root, file)
    #             solc_version = solidity.get_solc(full_path)
    #             if solc_version is None:
    #                 print('cannot get a correct solc version')
    #                 continue
    #             try:
    #                 scan_reentrancy = reentrancy_call(Slither(full_path, solc = solc_version))
    #                 print(scan_reentrancy.extract())
    #             except Exception:
    #                 print(f'compilation for {full_path} failed')

    # scan_reentrancy = reentrancy_call(Slither('test.sol'))
    # print(scan_reentrancy.extract())