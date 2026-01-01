# Static feature extraction for Python and C/C++ (heuristics)
import ast, re
from typing import Dict

def detect_language(filename: str) -> str:
    if filename.endswith('.py'):
        return 'python'
    if filename.endswith(('.cpp', '.cc', '.c', '.hpp', '.h')):
        return 'cpp'
    return 'unknown'

def features_from_python(code: str) -> Dict[str, float]:
    try:
        tree = ast.parse(code)
    except Exception:
        # unparsable -> return minimal features with penalty
        return {
            'num_lines': len(code.splitlines()),
            'num_functions': 0,
            'num_for': 0,
            'num_while': 0,
            'num_if': 0,
            'has_assert': 0,
            'avg_var_len': 0,
            'ast_depth': 0
        }

    num_lines = len(code.splitlines())
    num_functions = sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
    num_for = sum(isinstance(n, ast.For) for n in ast.walk(tree))
    num_while = sum(isinstance(n, ast.While) for n in ast.walk(tree))
    num_if = sum(isinstance(n, ast.If) for n in ast.walk(tree))
    has_assert = 1 if 'assert ' in code or 'assert(' in code else 0
    var_names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
    avg_var_len = sum(len(v) for v in var_names) / len(var_names) if var_names else 0

    def depth(node):
        if not isinstance(node, ast.AST):
            return 0
        maxd = 0
        for field in getattr(node, '_fields', []):
            val = getattr(node, field)
            maxd = max(maxd, depth(val))
        return 1 + maxd

    try:
        ast_depth = depth(tree)
    except Exception:
        ast_depth = 0

    return {
        'num_lines': num_lines,
        'num_functions': num_functions,
        'num_for': num_for,
        'num_while': num_while,
        'num_if': num_if,
        'has_assert': has_assert,
        'avg_var_len': avg_var_len,
        'ast_depth': ast_depth
    }

def features_from_cpp(code: str) -> Dict[str, float]:
    num_lines = len(code.splitlines())
    num_includes = len(re.findall(r'#include\b', code))

    # Very simple function heuristic: return_type name(
    num_functions = len(
        re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(', code)
    )

    num_for = len(re.findall(r'\bfor\s*\(', code))
    num_while = len(re.findall(r'\bwhile\s*\(', code))
    num_if = len(re.findall(r'\bif\s*\(', code))
    has_assert = 1 if 'assert(' in code or 'assert ' in code else 0

    return {
        'num_lines': num_lines,
        'num_includes': num_includes,
        'num_functions': num_functions,
        'num_for': num_for,
        'num_while': num_while,
        'num_if': num_if,
        'has_assert': has_assert,
    }
