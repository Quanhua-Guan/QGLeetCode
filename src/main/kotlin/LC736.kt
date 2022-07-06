import java.lang.StringBuilder
import java.util.*

class LC736 {
    /// 736. Lisp 语法解析
    class Solution01 {
        val scope = mutableMapOf<String, Deque<Int>>()
        var start = 0

        fun evaluate(expression: String): Int {
            return innerEvaluate(expression)
        }

        fun innerEvaluate(expression: String): Int {
            if (expression[start] != '(') { // 非表达式, 可能是整数或变量
                if (expression[start].isLowerCase()) { // 变量
                    val variable = parseVar(expression)
                    return scope[variable]!!.peek()
                } else { // 整数
                    return parseInt(expression)
                }
            }

            var ret = 0
            start++ // 移除左括号

            if (expression[start] == 'l') { // let
                start += 4 // 移除 "let "
                val variables = mutableListOf<String>()
                while (true) {
                    if (!expression[start].isLowerCase()) {
                        ret = innerEvaluate(expression) // let 表达式的最后一个 expr 表达式的值
                        break
                    }
                    val variable = parseVar(expression)
                    if (expression[start] == ')') {
                        ret = scope[variable]!!.peek() // let 表达式的最后一个 expr 表达式(变量)的值
                        break
                    }
                    variables.add(variable)
                    start++ // 移除空格
                    val e = innerEvaluate(expression)
                    scope.putIfAbsent(variable, ArrayDeque<Int>())
                    scope[variable]!!.push(e)
                    start++ // 移除空格
                }
                for (variable in variables) {
                    scope[variable]!!.pop() // 清除当前作用域的变量
                }
            } else if (expression[start] == 'a') { // add
                start += 4
                val e1 = innerEvaluate(expression)
                start++
                val e2 = innerEvaluate(expression)
                ret = e1 + e2
            } else if (expression[start] == 'm') { // mult
                start += 5
                val e1 = innerEvaluate(expression)
                start++
                val e2 = innerEvaluate(expression)
                ret = e1 * e2
            } else {
                assert(false)
            }

            start++ // 移除右括号
            return ret
        }

        fun parseInt(expression: String): Int {
            val n = expression.length
            var ret = 0
            var sign = 1
            if (expression[start] == '-') {
                sign = -1
                start++
            }
            while (start < n && expression[start].isDigit()) {
                ret = ret * 10 + (expression[start] - '0')
                start++
            }
            return sign * ret
        }

        fun parseVar(expression: String): String {
            val n = expression.length
            val ret = StringBuffer()

            while (start < n && expression[start] != ' ' && expression[start] != ')') {
                ret.append(expression[start])
                start++
            }

            return ret.toString()
        }
    }

    class Solution {
        val scope = mutableMapOf<String, Deque<Int>>()
        var start = 0

        fun evaluate(expression: String): Int {
            return innerEvaluate(expression)
        }

        fun innerEvaluate(expression: String): Int {
            if (expression[start] != '(') { // 非表达式, 可能是变量或数字
                if (expression[start].isLowerCase()) {  // 变量
                    val variable = parseVar(expression)
                    return scope[variable]!!.peek()
                } else { // 数字
                    return parseInt(expression)
                }
            }

            var ret = 0
            start++ // 跳过左括号

            if (expression[start] == 'l') { // let
                start += 4 // 跳过 "let "
                var variables = mutableListOf<String>()
                while (true) {
                    if (!expression[start].isLowerCase()) {
                        ret = innerEvaluate(expression)
                        break
                    }
                    val variable = parseVar(expression)
                    if (expression[start] == ')') {
                        ret = scope[variable]!!.peek()
                        break
                    }
                    variables.add(variable)
                    start++
                    val e = innerEvaluate(expression)
                    scope.putIfAbsent(variable, ArrayDeque<Int>())
                    scope[variable]!!.push(e)
                    start++
                }
                for (variable in variables) {
                    scope[variable]!!.pop()
                }
            } else if (expression[start] == 'a') { // add
                start += 4 // 跳过 "add "
                val e1 = innerEvaluate(expression)
                start++ // 跳过空格
                val e2 = innerEvaluate(expression)
                ret = e1 + e2
            } else if (expression[start] == 'm') { // mult
                start += 5 // 跳过 "mult "
                val e1 = innerEvaluate(expression)
                start++ // 跳过空格
                val e2 = innerEvaluate(expression)
                ret = e1 * e2
            } else {
                assert(false)
            }

            start++ // 跳过右括号
            return ret
        }

        fun parseInt(expression: String): Int {
            val n = expression.length
            var ret = 0
            var sign = 1

            if (expression[start] == '-') {
                sign = -1
                start++
            }

            while (start < n && expression[start].isDigit()) {
                ret = ret * 10 + (expression[start] - '0')
                start++
            }

            return ret * sign
        }

        fun parseVar(expression: String): String {
            val n = expression.length
            val variable = StringBuilder()
            while (start < n && expression[start] != ' ' && expression[start] != ')') {
                variable.append(expression[start])
                start++
            }
            return variable.toString()
        }
    }
}