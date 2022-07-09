fun main() {
    while (true) {
        try {
            println(
                LC873.Solution()
                    .lenLongestFibSubseq(intArrayOf(2, 4, 7, 8, 9, 10, 14, 15, 18, 23, 32, 50))
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}

class LC873 {
    /// 873. 最长的斐波那契子序列的长度
    class Solution {
        fun lenLongestFibSubseq(arr: IntArray): Int {
            val n = arr.size
            val numberSet = mutableSetOf<Int>()
            for (a in arr) {
                numberSet.add(a)
            }

            // 从 i 开始的目标序列长度
            var max = Int.MIN_VALUE
            for (i in 0 until n) {
                for (j in i + 1 until n) {
                    var a = arr[i]
                    var b = arr[j]
                    var c = a + b
                    var count = 2
                    while (numberSet.contains(c)) {
                        a = b
                        b = c
                        c = a + b
                        count += 1
                    }
                    max = maxOf(max, count)
                }
            }

            return if (max <= 2) 0 else max
        }
    }
}