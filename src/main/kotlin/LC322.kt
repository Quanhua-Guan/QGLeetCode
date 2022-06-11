class LC322 {
    /// 322. 零钱兑换
    class Solution322 {
        fun coinChange(coins: IntArray, amount: Int): Int {
            // 凑成 remain 数额需要使用的最小硬币数
            // mem[i] = 使用 coins 面额凑成 i 的最小硬币个数
            fun coinChange(remain: Int, mem: IntArray): Int {
                if (remain < 0) return -1
                if (remain == 0) return 0
                if (mem[remain] != 0) return mem[remain]

                var min = Int.MAX_VALUE
                for (c in coins) {
                    // 选择1个c
                    val count = coinChange(remain - c, mem)
                    if (count >= 0 && count < min) {
                        min = count + 1;
                    }
                }
                return (if (min == Int.MAX_VALUE) -1 else min).also { mem[remain] = it }
            }
            return coinChange(amount, IntArray(amount + 1))
        }
    }
}