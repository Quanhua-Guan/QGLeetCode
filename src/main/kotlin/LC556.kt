fun main() {
    while (true) {
        try {
            println(
                LC556.Solution().nextGreaterElement(230241)
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}

class LC556 {
    class Solution {
        fun nextGreaterElement(N: Int): Int {
            val digitList = mutableListOf<Int>()
            var n = N
            while (n > 0) {
                digitList.add(n % 10)
                n /= 10
            }

            val digits = digitList.toIntArray()
            var bound = -1
            outer@ for (r in 1 until digits.size) {
                for (l in 0 until r) {
                    if (digits[l] > digits[r]) {
                        bound = r
                        digits[l] = digits[r].also { digits[r] = digits[l] }
                        break@outer
                    }
                }
            }

            if (bound == -1) return -1

            val counts = IntArray(10)
            for (i in 0 until bound) {
                counts[digits[i]]++
            }
            var j = 0
            for (i in 9 downTo 0) {
                var count = counts[i]
                while (count > 0) {
                    digits[j++] = i
                    count--
                }
            }

            var res: Long = 0
            for (i in digits.size - 1 downTo 0) {
                res = res * 10 + digits[i]
            }
            if (res < Int.MIN_VALUE.toLong() || res > Int.MAX_VALUE.toLong()) return -1
            return res.toInt()
        }
    }
}