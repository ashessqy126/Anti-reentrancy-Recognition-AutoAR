import os,re,sys
from pathlib import Path
from typing import Optional
import solcx

def _get_os_name() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "macosx"
    if sys.platform == "win32":
        return "windows"
    raise OSError(f"Unsupported OS: '{sys.platform}' - py-solc-x supports Linux, OSX and Windows")

solcx.set_target_os(_get_os_name())

VOID_START = re.compile('//|/\*|"|\'')
PRAGMA = re.compile('pragma solidity.*?;')
QUOTE_END = re.compile("(?<!\\\\)'")
DQUOTE_END = re.compile('(?<!\\\\)"')

def remove_void(line):
    while True:
        m = VOID_START.search(line)
        if m is None:
            break
        if m[0] == '//':
            return (line[:m.start()], False)
        if m[0] == '/*':
            end = line.find('*/', m.end())
            if end == -1:
                return (line[:m.start()], True)
            else:
                line = line[:m.start()] + line[end+2:]
                continue
        if m[0] == '"':
            m2 = DQUOTE_END.search(line[m.end():])
        else: # m[0] == "'":
            m2 = QUOTE_END.search(line[m.end():])
        if m2:
            line = line[:m.start()] + line[m.end()+m2.end():]
            continue
        # we should not arrive here for a correct Solidity program
        return (line[:m.start()], False)
    return (line, False)

def get_pragma(file: str) -> Optional[str]:
    in_comment = False
    for line in file.splitlines():
        if in_comment:
            end = line.find('*/')
            if end == -1:
                continue
            else:
                line = line[end+2:]
        line, in_comment = remove_void(line)
        m = PRAGMA.search(line)
        if m:
            return m[0]
    return None

def get_pragmas(file: str) -> set:
    pragmas = set()
    in_comment = False
    for line in file.splitlines():
        if in_comment:
            end = line.find('*/')
            if end == -1:
                continue
            else:
                line = line[end+2:]
        line, in_comment = remove_void(line)
        m = PRAGMA.search(line)
        if m:
            pragmas.add(m[0])
    return pragmas

def get_solc(filename: str) -> Optional[Path]:
    with open(filename) as f:
        file = f.read()
    try:
        pragmas = get_pragmas(file)
        new_pragmas = set()
        for pragma in pragmas:
            new_pragmas.add(pragma)
        version = solcx.install_solc_pragma_by_set(new_pragmas)
        return solcx.get_executable(version)
    except Exception as e:
        return None
