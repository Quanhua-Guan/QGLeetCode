class LC1043 {
    /// 1403. 非递增顺序的最小子序列
    class Solution {
        fun minSubsequence(nums: IntArray): List<Int> {
            nums.sortDescending()
            var sum = nums.sum()
            var _sum = 0
            val result = mutableListOf<Int>()
            var i = 0
            while (i < nums.size && _sum <= sum - _sum) {
                _sum += nums[i]
                result.add(nums[i])
                i++
            }
            return result
        }
    }
}