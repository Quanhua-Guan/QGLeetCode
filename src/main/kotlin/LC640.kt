class LC640 {
    /// 640. 求解方程
    class Solution {
        fun solveEquation(equation: String): String {
            var leftXCount = 0
            var leftNumber = 0
            var rightXCount = 0
            var rightNumber = 0

            /// 扫描字符串, 取出每一个元素 (数字 或 变量)
            var equalSignScanned = false // 是否已扫描到符号'=', 用于区分表达式左/右半部分.
            var sign = 1 // 当前元素正负 +1/-1
            var num = 0  // 当前元素数字部分
            var numFilled = false // 当前元素是否已经读取过数字, 默认 false
            var isX = false       // 当前元素是否是变量, 默认 false

            var i = 0
            while (i <= equation.length) {
                // 此处相当于在表达式最后添加了一个 '+', 用作结束标志.
                var c = if (i == equation.length) '+' else equation[i]

                if (c == '+' || c == '-' || c == '=') {
                    if (equalSignScanned) {
                        // right
                        if (isX) {
                            rightXCount += sign * num
                        } else {
                            rightNumber += sign * num
                        }
                    } else {
                        // left
                        if (isX) {
                            leftXCount += sign * num
                        } else {
                            leftNumber += sign * num
                        }
                    }

                    // 准备识别下一个元素 (数字 或 变量)
                    isX = false
                    num = 0
                    numFilled = false
                    sign = if (c == '-') -1 else 1

                    // 区分表达式 左半部分 和 右半部分
                    if (c == '=') equalSignScanned = true
                } else if (c == 'x') {
                    isX = true
                    // 判断是否已经读取过数字, 没有则 num 为 0, 需要主动置为 1
                    // 根据题意, 此处不需要标记 numFilled = true, 因为读到 'x' 说明当前变量已经读完, 后续将开始读取下一个元素.
                    if (!numFilled) num = 1
                } else {
                    assert(c.isDigit())
                    num = num * 10 + (c - '0')
                    numFilled = true // 标记已经读取过数字
                }

                i++
            }

            // 化简成 leftXCount * x = rightNumber 的形式
            leftXCount -= rightXCount
            rightNumber -= leftNumber

            // 对 leftXCount 和 rightNumber 分情况讨论
            if (leftXCount == 0) {
                if (rightNumber == 0) {
                    // 0x = 0, 所以 x 取任意数都行
                    return "Infinite solutions"
                } else {
                    // 0x = [非0数] => 0 == [非0数], 显然逻辑冲突, 所以无解
                    return "No solution"
                }
            } else {
                // 此时, leftXCount != 0
                if (rightNumber == 0) {
                    // leftXCount * x = 0 => 所以, x = 0
                    return "x=0"
                } else {
                    // leftXCount * x = rightNumber => 所以, x = rightNumber / leftXCount (注意: 此处除法丢弃了余数)
                    return "x=" + (rightNumber / leftXCount).toString()
                }
            }
        }
    }
}