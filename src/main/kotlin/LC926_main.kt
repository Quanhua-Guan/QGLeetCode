fun main() {
    while (true) {
        try {
            println(
                Solution().minFlipsMonoIncr("00011000")
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}

/// 926. 将字符串翻转到单调递增
class Solution {
    fun minFlipsMonoIncr(s: String): Int {
        // 转成求解s中最长单调递增[子序列]的长度

        // dp[i]代表长度为i的单调递增子序列的最小末尾字符
        val dp = CharArray(s.length + 1)
        dp[0] = 0.toChar()
        var len = 1
        dp[len] = s[0]
        for (i in 1 until s.length) {
            if (s[i] >= dp[len]) {
                dp[++len] = s[i]
            } else {
                var l = 0
                var r = len
                val target = s[i] // s[i] < dp[len]
                var pos = 0
                // 在 dp[1..len] 中找到最后一个小于等于 target 的字符对应的下标
                while (l <= r) {
                    val m = (l + r) ushr 1
                    if (dp[m] <= target && (m == len || dp[m + 1] > target)) {
                        pos = m
                        break
                    }
                    if (dp[m] <= target) {
                        l = m + 1
                    } else {
                        r = m - 1
                    }
                }
                dp[pos + 1] = target
            }
        }
        return s.length - len
    }
}
