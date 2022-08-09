class LC1413 {
    /// 1413. 逐步求和得到正数的最小值
    class Solution {
        fun minStartValue(nums: IntArray): Int {
            var startValue = 1
            var gotLessThan1Value = false
            for (i in nums.size - 1 downTo 0) {
                val n = nums[i]

                if (!gotLessThan1Value && n < 1) {
                    gotLessThan1Value = true
                }

                if (gotLessThan1Value) {
                    startValue = maxOf(1, startValue - n)
                }

            }
            return maxOf(1, startValue)
        }
    }
}