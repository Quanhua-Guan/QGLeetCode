class LC494 {
    /// 494. 目标和
    class Solution_Recall {
        fun findTargetSumWays(nums: IntArray, target: Int): Int {
            val n = nums.size
            var count = 0

            fun search(index: Int, sum: Int) {
                if (index == n) {
                    if (sum == target) count++
                    return
                }

                search(index + 1, sum + nums[index])
                search(index + 1, sum - nums[index])
            }

            search(0, 0)
            return count
        }
    }

    class Solution_DP {
        fun findTargetSumWays(nums: IntArray, target: Int): Int {
            val n = nums.size

            /// 问题: 从 nums 中选出若干个数填上-号, 其余填上+号, 并让最终表达式和为target
            ///
            /// 假设 sum 为 nums 总和, 选择其中若干个数填+号, 剩余若干个数填-号. 假设填+号的数总和为 positive,
            /// 则所求目标为: positive - (sum - positive) = target => 2 * positive - sum = target => positive = (sum + target) / 2
            ///
            /// 将问题转化为: 从 nums 中选出若干个数让其和为 positive 的方案数
            ///
            val sum = nums.sum()
            var positive = sum + target
            if (positive < 0 || positive and 1 == 1) return 0 // sum + target 显然应该大于等于0, 且 sum + target 必须是偶数才可能被2整除.
            positive /= 2

            // dp[i][j] 从 nums 前 i 个数中选出若干个数总和为 j 的方案数
            // dp[0][0] = 1
            // dp[0][j] = 0, j in 1..positive
            // 所求结果为 dp[n][positive]
            // dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i]] or dp[i - 1][j]
            val dp = Array(n + 1) { IntArray(positive + 1) }
            dp[0][0] = 1

            for (i in 1..n) { // 考察前i个数
                for (j in 0..positive) { // 考察可能的和. 目标: 更新 dp[i][j]
                    dp[i][j] = dp[i - 1][j]
                    if (j - nums[i - 1] >= 0) { // j - nums[i - 1] >= 0 才满足下标要求, i - 1 对应前 i 个数的最后一位数.
                        dp[i][j] += dp[i - 1][j - nums[i - 1]]
                    }
                }
            }

            return dp[n][positive]
        }
    }
}