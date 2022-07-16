class JZO14I {
    /// 剑指 Offer 14- I. 剪绳子
    class Solution_recur {
        val mem = mutableMapOf<Int, Int>()
        fun cuttingRope(n: Int): Int {
            if (n == 1 || n == 2) return 1
            if (n == 3) return 2
            if (n == 4) return 4

            if (mem.containsKey(n)) return mem[n]!!

            val max = maxOf(
                maxOf(2 * (n - 2), 2 * cuttingRope(n - 2)),
                maxOf(3 * (n - 3), 3 * cuttingRope(n - 3))
            )
            return max.also { mem[n] = it }
        }
    }

    class Solution_dp {
        fun cuttingRope(n: Int): Int {
            val dp = IntArray(n + 1)
            dp[2] = 1
            for (i in 3..n) {
                dp[i] = maxOf(maxOf(2 * (i - 2), 2 * dp[i - 2]), maxOf(3 * (i - 3), 3 * dp[i - 3]))
            }
            return dp[n]
        }
    }

    class Solution_math {
//        public int cuttingRope(int n) {
//            if(n <= 3) return n - 1;
//            int a = n / 3, b = n % 3;
//            if(b == 0) return (int)Math.pow(3, a);
//            if(b == 1) return (int)Math.pow(3, a - 1) * 4;
//            return (int)Math.pow(3, a) * 2;
//        }
        fun cuttingRope(n: Int): Int {
            if (n <= 3) return n - 1
            val a = n / 3
            val b = n % 3
            if (b == 0) return Math.pow(3.0, a.toDouble()).toInt()
            if (b == 1) return (Math.pow(3.0, a - 1.0) * 4.0).toInt()
            return (Math.pow(3.0, a.toDouble()) * 2).toInt()
        }
    }
}