fun main() {
    while (true) {
        try {
            println(
                LC592.Solution().fractionAddition("1/2-1/2")
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}

class LC592 {
    /// 592. 分数加减运算
    class Solution {
        // -1/2+1/2+1/3
        fun fractionAddition(expression: String): String {
            var i = 0
            fun _parseInt(): Int {
                var sign = 1
                var res = 0
                if (expression[i] == '-') {
                    sign = -1
                    i++
                }
                while (i < expression.length && expression[i].isDigit()) {
                    res = res * 10 + (expression[i] - '0')
                    i++
                }

                return sign * res
            }
            if (i < expression.length && expression[i] == '+') i++ // 跳过首个 '+' 字符(如果存在)

            val numbers = mutableListOf<Pair<Int, Int>>()
            var sign = 1;
            while (i < expression.length) {
                var numerator = _parseInt() * sign // 解析 分子
                i++                                // 跳过 '/'
                var denominator = _parseInt()      // 解析 分母
                numbers.add(Pair(numerator, denominator)) // 以Pair(分子,分母)方式存下来

                if (i == expression.length) break; // 判断是否到了表达式字符串末尾

                if (expression[i] == '+') {    // 解析操作符 +-
                    sign = 1
                } else { // expression[i] == '+'
                    sign = -1
                }
                i++
            }

            var (n, d) = numbers[0]
            for (i in 1 until numbers.size) {
                val (_n, _d) = numbers[i]
                n = n * _d + _n * d
                d = d * _d
                if (n != 0 && d != 0) {
                    val g = gcd(n, d)
                    n /= g
                    d /= g
                }
            }

            if (n == 0) d = 1
            if (n > 0 && d < 0) {
                n = -n
                d = -d
            }
            return "$n/$d"
        }

        fun gcd(a: Int, b: Int): Int {
            if (b == 0) return a
            return gcd(b, a % b)
        }
    }
}
