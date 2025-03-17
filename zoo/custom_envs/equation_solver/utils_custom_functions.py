from sympy import simplify, factor, sqrt, Pow, Basic, Integer, Mul, Add, collect, together
from sympy import Eq, ratsimp, expand, Abs, Rational, Number, sympify, symbols, integrate
from sympy import sin, cos, tan, asin, acos, atan, Symbol
from operator import add, sub, mul, truediv

def custom_identity(expr, term):
    return expr

def custom_expand(expr, term):
    return expand(expr)

def custom_simplify(expr, term):
    return simplify(expr)

def custom_factor(expr, term):
    return factor(expr)

def custom_collect(expr, term):
    return collect(expr, term)

def custom_together(expr, term):
    return together(expr)

def custom_ratsimp(expr, term):
    return ratsimp(expr)

def custom_square(expr, term):
    return expr**2

def custom_sqrt(expr, term):
    # Check if the expression is a perfect square
    simplified_expr = simplify(expr)

    # Case 1: If it's a square of a single term (like x**2), return the term
    if simplified_expr.is_Pow and simplified_expr.exp == 2:
        base = simplified_expr.base
        return base

    # Case 2: Otherwise, return ±sqrt(expression)
    return sqrt(expr)

def inverse_sin(expr, term):
    if isinstance(expr, (int, float)):
        return asin(expr)
    if expr.has(sin):
        return expr.replace(
            lambda arg: arg.func == sin,
            lambda arg: arg.args[0]
        )
    return asin(expr)

def inverse_cos(expr, term):
    if isinstance(expr, (int, float)):
        return acos(expr)
    if expr.has(cos):
        return expr.replace(
            lambda arg: arg.func == cos,
            lambda arg: arg.args[0]
        )
    return acos(expr)

def inverse_tan(expr, term):
    if isinstance(expr, (int, float)):
        return atan(expr)
    if expr.has(tan):
        return expr.replace(
            lambda arg: arg.func == tan,
            lambda arg: arg.args[0]
        )
    return atan(expr)


operation_names = {
    add: "add",
    sub: "subtract",
    mul: "multiply",
    truediv: "divide",
    custom_expand: "expand",
    custom_simplify: "simplify",
    custom_factor: "factor",
    custom_collect: "collect",
    custom_together: "together",
    custom_ratsimp: "ratsimp",
    custom_square: "square",
    custom_sqrt: "sqrt",
    inverse_sin: 'sin^{-1}',
    inverse_cos: 'cos^{-1}',
    inverse_tan: 'tan^{-1}',
    custom_identity: 'identity'
}
