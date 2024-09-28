####################
# Filename: requirement_A2.py

# This file should be saved in the same folder as A2.py
# and should not be modified.
# Source code in this file is partially adapted from python 
# linters (inspired by compile-time program analysis).
####################

import sys

(ver, subver, micro, releaselevel, serial) = sys.version_info
if ((ver < 3) or (subver < 7 or subver >= 10)):
    raise Exception("Please use Python version between 3.7 and 3.9.")

import parser, traceback, inspect, platform
import math

_banned = ('qutip, qiskit, eval, exec, __import__')



def _writeErrorMsg(file, line, fn, text, msg):
    messages = '\n'
    if (file): messages += '  File "%s", ' % file
    if (line): messages += 'line %d,' % line
    if (fn): messages += 'in %s' % fn
    if (text): messages += '\n     %s' % text.strip()
    messages += '\n  Error:    %s' % msg
    return messages

class _CheckError(Exception):
    def __init__(self, errors):
        messages = [ '' ]
        for i,e in enumerate(errors):
            (msg, file, line, fn, text) = e
            message = _writeErrorMsg(file, line, fn, text, msg)
            messages.append(message)
        message = ''.join(messages)
        super().__init__(message)

class _Checker(object):
    def __init__(self, code=None, filename=None, banned=None):
        self.code = code
        self.filename = filename
        self.banned = set(banned or [ ])

    def report(self, msg, line=None, fn=None, text=None, node=None):
        if (node != None) and (type(node) in (list, tuple)):
            (nodeTid, nodeText, nodeLine, nodeCol) = node
            line = nodeLine
        if ((text == None) and
            (line != None) and
            (1 <= line <= len(self.lines))):
            text = self.lines[line-1]
        self.errors.append((msg, self.filename, line, fn, text))

    def checkRoot(self):
        # Allowed: def, class, main, import, from import
        for rootNodes in self.astList:
            if (not isinstance(rootNodes, list)):
                msg = 'Parsed AST Error.'
                self.report(msg, node=rootNode)
            rootNode = rootNodes[0]
            if (isinstance(rootNode, int)):
                if (rootNode == 3):
                    text = 'root'
            elif isinstance(rootNode, list) and \
                isinstance(rootNode[0], list) and \
                len(rootNode[0]) == 4 and rootNode[0][1] == "@":
                (tid, text, line, col) = rootNode[0]
            elif ((type(rootNode) not in [list,tuple]) or
                  (len(rootNode) != 4)):
                msg = 'Unknown root function: %r' % rootNode
                self.report(msg)
                continue
            else:
                (tid, text, line, col) = rootNode
            if (text not in ['@', 'import', 'from', 'def',
                             'class', 'np', 'root']):
                msg = "Only import, def, or class are allowed."
                self.report(msg, node=rootNode)

    def checkAll(self, astList):
        if (isinstance(astList[0], list)):
            for node in astList: self.checkAll(node)
        else:
            node = astList
            (tid, text, line, col) = node
            if (text in self.banned):
                msg = 'Disallowed: "%s"' % text
                self.report(msg, node=node)            

    def check(self):
        print('Checking... ', end='')
        self.errors = [ ]
        if (self.code == None):
            with open(self.filename, 'rt', encoding="utf-8") as f:
                try: self.code = f.read()
                except e:
                    msg = 'Error when trying to read file:\n' + str(e)
                    self.report(msg)
                    raise _CheckError(self.errors)
        if (self.code in [None,'']):
            self.report('Could not read code from "%s"' % self.filename)
            raise _CheckError(self.errors)
        self.lines = self.code.splitlines()
        self.st = parser.suite(self.code)
        self.stList = parser.st2list(self.st, line_info=True, col_info=True)
        self.astList = self.parseAST(self.stList, plain=False)
        self.astPlainList = self.parseAST(self.stList, plain=True)
        if (self.astPlainList[-1] in [
            ['if', ['__name__', '==', "'__main__'"],
                   ':', ['main', ['(', ')']]],
            ['if', ['(', ['__name__', '==', "'__main__'"], ')'],
                   ':', ['main', ['(', ')']]],
            ['if', ['__name__', '==', '"__main__"'],
                   ':', ['main', ['(', ')']]],
            ['if', ['(', ['__name__', '==', '"__main__"'], ')'],
                   ':', ['main', ['(', ')']]]
            ]):
            self.astPlainList.pop()
            self.astList.pop()
        self.checkRoot()
        self.checkAll(self.astList)
        if (self.errors != [ ]):
            raise _CheckError(self.errors)
        print("Passed!")

    def parseAST(self, ast, plain):
        if (not isinstance(ast, list)): return None
        if (not isinstance(ast[1], list)):
            result = ast[1]
            if (result == ''): result = None
            if ((not plain) and (result != None)): result = ast
            return result
        result = [ ]
        for val in ast:
            node = self.parseAST(val, plain)
            if (node != None):
                result.append(node)
        if (len(result) == 1): result = result[0]
        return result

def check(code=None, filename=None, banned=_banned):
    if (isinstance(banned, str)):
        banned = banned.split(',')
    if ((code == None) and (filename == None)):
        try:
            module = None
            frame = inspect.stack()[1]
            module = inspect.getmodule(frame[0])
            if ((module == None) or (module.__file__ == None)):
                # this may help, or maybe not (sigh)
                module = sys.modules['__main__']
            # the next line may fail (sigh)
            filename = module.__file__
        except:
            raise Exception('check cannot find module/file to check!')
    try:
        _Checker(code=code, filename=filename, banned=banned).check()
    except _CheckError as checkError:
        # just 'raise checkError' for cleaner traceback
        checkError.__traceback__ = None
        raise checkError

def _versionCheck():
    (ver, subver, micro, releaselevel, serial) = sys.version_info
    if ((ver < 3) or (subver < 7 or subver >= 10)):
        raise Exception("Please use Python version between 3.7 and 3.9.")

class _AssertionError(AssertionError): pass

if (__name__ != '__main__'):
    _versionCheck()
