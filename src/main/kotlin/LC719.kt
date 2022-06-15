class LC719 {
    /// 719. 找出第 K 小的数对距离
    class Solution {
        fun bs(nums: IntArray, target: Int, end: Int): Int {
            var l = 0
            var r = end
            while (l <= r) {
                val m = (l + r) ushr 1
                if (nums[m] >= target && (m == 0 || nums[m - 1] < target)) {
                    // 找到第一个大于等于target的数的下标
                    return m
                } else if (nums[m] >= target) {
                    r = m - 1
                } else {
                    l = m + 1
                }
            }
            return -1
        }
        fun smallestDistancePair(nums: IntArray, k: Int): Int {
            nums.sort()
            val n = nums.size
            var min = 0
            var max = nums[n - 1] - nums[0] // 最大距离
            while (min <= max) {
                val mid = (min + max) ushr 1
                var count = 0 // 计算以 j 结尾的数对中距离小于 mid 的个数
                for (j in 0 until n) {
                    val i = bs(nums, nums[j] - mid, j) // 一定可以找到 i, 因为有 nums[j] 兜底.
                    count += j - i
                }
                if (count >= k) {
                    // 太大, 上界往左移动
                    max = mid - 1
                } else {
                    min = mid + 1
                }
            }
            return min
        }
    }
}