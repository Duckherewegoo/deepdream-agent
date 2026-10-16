"""科学计算器工具"""
import math
import json
from langchain_core.tools import tool


@tool
def scientific_calculator(expression: str) -> str:
    """科学计算器，支持复杂数学运算。

    支持的运算：
    - 基本运算：加减乘除 (+, -, *, /)
    - 幂运算：** 或 pow()
    - 三角函数：sin, cos, tan, asin, acos, atan（弧度）
    - 对数：log(自然对数), log10, log2
    - 其他：sqrt, abs, ceil, floor, round
    - 常数：pi, e

    Args:
        expression: 数学表达式字符串，如 "sin(pi/2) + sqrt(16)"

    Returns:
        计算结果的 JSON 字符串
    """
    try:
        # 安全的命名空间，只允许数学相关函数和常量
        safe_namespace = {
            # 常量
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau,
            'inf': math.inf,
            'nan': math.nan,

            # 基本函数
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,

            # 幂运算与开方
            'sqrt': math.sqrt,
            'pow': math.pow,
            'exp': math.exp,

            # 对数
            'log': math.log,      # 自然对数
            'log10': math.log10,  # 常用对数
            'log2': math.log2,    # 二进制对数

            # 三角函数（弧度）
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'atan2': math.atan2,

            # 双曲函数
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,

            # 取整
            'ceil': math.ceil,
            'floor': math.floor,
            'trunc': math.trunc,

            # 其他
            'fabs': math.fabs,
            'factorial': math.factorial,
            'gcd': math.gcd,
            'hypot': math.hypot,
            'degrees': math.degrees,  # 弧度转角度
            'radians': math.radians,  # 角度转弧度
        }

        # 计算表达式
        result = eval(expression, {"__builtins__": {}}, safe_namespace)

        return json.dumps({
            "success": True,
            "expression": expression,
            "result": result,
            "detail": f"计算结果：{expression} = {result}"
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "expression": expression,
            "error": str(e),
            "detail": f"计算错误：{str(e)}"
        }, ensure_ascii=False)
