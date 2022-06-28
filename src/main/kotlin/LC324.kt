class LC324 {
    // 324. 摆动排序 II
    class Solution {
        fun wiggleSort(nums: IntArray): Unit {
            val tmp = nums.clone()
            tmp.sort()

            val n = nums.size
            val p = (n + 1) / 2

            var i = 0
            var j = p - 1
            var k = n - 1
            while (i < n) {
                nums[i] = tmp[j]
                if (i + 1 < n) {
                    nums[i + 1] = tmp[k]
                }

                i += 2
                j -= 1
                k -= 1
            }
        }
    }
}
