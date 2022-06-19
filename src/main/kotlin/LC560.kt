class LC560 {
    /// 560. 和为 K 的子数组
    class Solution_Vio {
        fun subarraySum(nums: IntArray, k: Int): Int {
            val n = nums.size
            var count = 0
            for (start in 0 until n) { // 枚举起点 start
                var sum = 0
                for (end in start until n) { // 枚举终点 end
                    sum += nums[end]
                    if (sum == k) {
                        count++
                    }
                }
            }
            return count
        }
    }

    /// 560. 和为 K 的子数组
    class Solution_PreSum {
        fun subarraySum(nums: IntArray, k: Int): Int {
            val n = nums.size
            var count = 0
            var preSum = 0

            val sumCountMap = mutableMapOf<Int, Int>()
            // 空数组也是1个子数组
            sumCountMap[0] = 1

            for (i in 0 until n) { // 枚举起点 start
                preSum += nums[i]
                if (sumCountMap.containsKey(preSum - k)) {
                    count += sumCountMap[preSum - k]!!
                }
                sumCountMap[preSum] = sumCountMap.getOrDefault(preSum, 0) + 1
            }
            return count
        }
    }
}