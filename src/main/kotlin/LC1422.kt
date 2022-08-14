class LC1422 {
    /// 1422. 分割字符串的最大得分
    class Solution {
        fun maxScore(s: String): Int {
            var cnt0 = 0
            var cnt1 = 0
            var totalCnt1 = 0

            for (c in s) {
                if (c == '1') totalCnt1++
            }

            var max = 0
            for (i in 0 until (s.length - 1)) {
                val c = s[i]
                if (c == '0') ++cnt0
                else ++cnt1

                max = maxOf(max, cnt0 + totalCnt1 - cnt1)
            }

            return max
        }
    }
}