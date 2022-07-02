/// 剑指 Offer 46. 把数字翻译成字符串
class JZO46 {
    class Solution {
        fun translateNum(num: Int): Int {
            if (num == 0) return 1

            val digitList = mutableListOf<Int>()
            var n = num
            while (n > 0) {
                digitList.add(n % 10)
                n /= 10
            }
            digitList.reverse()
            val digits = digitList.toIntArray()

            val countMemoryWhenPrevIndexNotUsed = IntArray(digits.size) { -1 }
            val countMemoryWhenPrevIndexUsed = IntArray(digits.size) { -1 }
            fun countFrom(index: Int, prevIndexUsed: Boolean): Int {
                if (index >= digits.size) {
                    if (prevIndexUsed) {
                        return 1
                    } else {
                        return 0
                    }
                }

                if (prevIndexUsed) {
                    if (countMemoryWhenPrevIndexUsed[index] != -1) return countMemoryWhenPrevIndexUsed[index]

                    val count = countFrom(index + 1, true) + countFrom(index + 1, false)
                    return count.also { countMemoryWhenPrevIndexUsed[index] = it }
                } else {
                    if (countMemoryWhenPrevIndexNotUsed[index] != -1) return countMemoryWhenPrevIndexNotUsed[index]

                    val prevN = digits[index - 1]
                    val n = digits[index]
                    var count = 0
                    if (prevN != 0 && prevN * 10 + n in 0..25) { /// 考虑前导 0 的特殊情况
                        count = countFrom(index + 1, true)
                    }
                    return count.also { countMemoryWhenPrevIndexNotUsed[index] = it }
                }
            }

            return countFrom(0, true)
        }
    }

    class Solution_DP {
        fun translateNum(num: Int): Int {
            if (num == 0) return 1

            val digitList = mutableListOf<Int>()
            var number = num
            while (number > 0) {
                digitList.add(number % 10)
                number /= 10
            }
            digitList.reverse()
            val digits = digitList.toIntArray()

            val n = digits.size
            /// count[i] 代表下标为 count[i..(n-1)] 翻译的方法数
            /// 显然 count[n - 1] = 1, 另外可以设置 count[n] = 1
            /// count[i] = count[i + 1] + (if(digits[i] != 0 && (digits[i] * 10 + digits[i + 1]) in 0..25) 1 else 0) * count[i + 2]
            /// 结果为 count[0]
            val count = IntArray(n + 1)
            count[n] = 1
            count[n - 1] = 1

            for (i in n - 2 downTo 0) {
                count[i] = count[i + 1]
                if (digits[i] != 0 && digits[i] * 10 + digits[i + 1] in 0..25) {
                    count[i] += count[i + 2]
                }
            }

            return count[0]
        }
    }
}