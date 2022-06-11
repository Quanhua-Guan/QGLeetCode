class LC583 {
    /// 583. 两个字符串的删除操作
    class Solution {
        fun minDistance(word1: String, word2: String): Int {
            val length = longestCommonSubsequence(word1, word2)
            return word1.length + word2.length - length * 2;
        }

        fun longestCommonSubsequence(text1: String, text2: String): Int {
            val n1 = text1.length
            val n2 = text2.length

            if (n1 < 1 || n2 < 1) return 0

            // dp[i][j]代表 text1[0..i] 和 text2[0..j] 代表最长公共子序列的长度
            // dp[0][0] = if (text1[0] == text2[0]) 1 else 0
            // dp[i][j] 最小值为0, 最大值为 minOf(i, j) + 1
            val dp = Array(n1) { IntArray(n2) }
            // 初始值
            dp[0][0] = if (text1[0] == text2[0]) 1 else 0
            // 根据初始值推导 dp[i][0] 和 dp[0][i]
            for (i in 1 until n1) {
                if (text1[i] == text2[0]) {
                    dp[i][0] = 1
                } else {
                    dp[i][0] = dp[i - 1][0]
                }
            }
            for (j in 1 until n2) {
                if (text1[0] == text2[j]) {
                    dp[0][j] = 1
                } else {
                    dp[0][j] = dp[0][j - 1]
                }
            }
            for (i in 1 until n1) {
                for (j in 1 until n2) {
                    if (text1[i] == text2[j]) {
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    } else {
                        dp[i][j] = maxOf(dp[i - 1][j], dp[i][j - 1])
                    }
                }
            }
            return dp[n1 - 1][n2 - 1]
        }
    }
}