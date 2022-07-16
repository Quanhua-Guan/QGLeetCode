class JZO62 {
    /// 剑指 Offer 62. 圆圈中最后剩下的数字
    class Solution1 {
        fun lastRemaining(N: Int, m: Int): Int {
            // 总共有 N 个数, [0..N-1]
            // 假设 N = 4, m = 3
            //
            // 第1次移除前(序列长度5): 0, 1, 2, 3, 4 => 第1次将移除 2
            // 第1次移除后(序列长度4): 3, 4, 0, 1 (第2次移除时从 3 开始算第 1 个) => 第2次将移除0
            //
            // 第2次移除后(序列长度3): 1, 3, 4
            // 第3次移除后(序列长度2): 1, 3, (1)
            // 第4次移除后(序列长度1): 3

            // f(n)代表序列长度为n时, 目标数字在当前序列数组中的下标.
            fun f(n: Int): Int {
                if (n == 1) return 0 // 显然, 如果序列长度为1, 则该序列的唯一数字即为目标数字, 下标为0
                val x = f(n - 1)
                return (x + m) % n
            }

            return f(N)
        }
    }

    class Solution_2 {
        fun lastRemaining(N: Int, m: Int): Int {
            var x = 0
            for (n in 2 until N + 1) {
                x = (x + m) % n
            }
            return x
        }
    }
}