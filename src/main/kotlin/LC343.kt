class LC343 {
    class Solution_Recall {
        fun integerBreak(n: Int): Int {
            // 最多可以将 n 拆成 n 个正整数, 最少 1 个 (根据题意限制最少是2个)
            // 每次将 n 拆成 n1 和 n2 两部分, 且令 integerBreak(n1) * integerBreak(n2) 最大化.
            fun integerBreak_(n: Int, mustBreak:Boolean, mem: IntArray): Int {
                if (n == 1) return 1
                if (mem[n] != 0) return mem[n]

                var product = if (mustBreak) 1 else n
                for (i in 1 until n) {
                    product = maxOf(product, integerBreak_(i, false, mem) * integerBreak_(n - i, false, mem))
                }
                return product.also { mem[n] = it }
            }

            val mem = IntArray(n + 1)
            return integerBreak_(n, true, mem)
        }
    }

    class Solution_DP {
        fun integerBreak(N: Int): Int {
            val maxProduct = IntArray(N)
            for (n in 1 until N) {
                var max = n
                for (i in 1 until n) {
                    max = maxOf(max, maxProduct[i] * maxProduct[n - i])
                }
                maxProduct[n] = max
            }

            var max = 1
            for (i in 1 until N) {
                max = maxOf(max, maxProduct[i] * maxProduct[N - i])
            }
            return max
        }
    }
}