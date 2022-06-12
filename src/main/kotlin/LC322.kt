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

                // dp[amount] = minOf{ dp[amount - c] + 1 }
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

    class Solution_20220612_Recall {
        fun coinChange(coins: IntArray, amount: Int): Int {
            // 做减法

            // 计算凑整剩余 remain 金额需要的最小硬币数 (记忆计算结果)
            fun coinChange(remain: Int, mem: IntArray): Int {
                if (remain < 0) return -1 // 无法凑整
                if (remain == 0) return 0

                if (mem[remain] != 0) return mem[remain]

                var min = Int.MAX_VALUE
                for (coin in coins) {
                    val count = coinChange(remain - coin, mem)
                    if (count >= 0) { // 有效凑整
                        min = minOf(min, count + 1)
                    }
                }

                return (if (min == Int.MAX_VALUE) -1 else min).also { mem[remain] = it }
            }
            return coinChange(amount, IntArray(amount + 1))
        }
    }

    /// 322. 零钱兑换
    class Solution322_DP {
        fun coinChange(coins: IntArray, amount: Int): Int {
            // 凑成 remain 数额需要使用的最小硬币数
            // quantity[i] = 使用 coins 面额凑成 i 的最小硬币个数
            // quantity[i] = minOf{ quantity[i - c] + 1 }
            val quantity = IntArray(amount + 1) { amount + 1 } // quantity[i] 最大值为 amount, 取 >=amount + 1 则说明无法凑整.
            quantity[0] = 0
            for (coinAmount in 1 until amount + 1) {
                for (coin in coins) {
                    if (coin <= coinAmount) {
                        quantity[coinAmount] = minOf(quantity[coinAmount], quantity[coinAmount - coin] + 1)
                    }
                }
            }
            if (quantity[amount] > amount) return -1
            return quantity[amount]
        }
    }

    class Solution_20220612_DP {
        fun coinChange(coins: IntArray, amount: Int): Int {
            coins.sort()
            // 凑成 remain 数额需要使用的最小硬币数
            // quantity[i] = 使用 coins 面额凑成 i 的最小硬币个数
            // quantity[i] = minOf{ quantity[i - c] + 1 }, 即: 找到 c 让 quantity[i - c] + 1 最小
            val quantity = IntArray(amount + 1) { amount + 1 } // quantity[i] 最大值为 amount, 取 >=amount + 1 则说明无法凑整.
            quantity[0] = 0
            for (coinAmount in 1 until amount + 1) { // 目标: 更新 quantity[coinAmount]
                for (coin in coins) {
                    if (coinAmount >= coin) {
                        quantity[coinAmount] = minOf(quantity[coinAmount], quantity[coinAmount - coin] + 1)
                    } else {
                        break
                    }
                }
            }
            if (quantity[amount] > amount) return -1
            return quantity[amount]
        }
    }

}