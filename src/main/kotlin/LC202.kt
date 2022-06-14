class LC202 {
    /// 202. 快乐数
    class Solution {
        fun next(n: Int): Int {
            var res = 0
            var n = n
            while (n > 0) {
                val r = n % 10
                res += r * r
                n = n / 10
            }
            return res
        }
        fun isHappy(n: Int): Boolean {
            var slow = n
            var fast = n
            while (true) {
                slow = next(slow)
                fast = next(next(fast))
                if (slow == 1) { // if2, 说明: 特例 n == 1 的时候, 这个 if1 要放 if2 前面
                    return true
                }
                if (slow == fast) { // if2
                    return false
                }
            }
            return true
        }
    }
}