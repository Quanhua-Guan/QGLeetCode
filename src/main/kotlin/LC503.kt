import java.util.*

class LC503 {
    /// 503. 下一个更大元素 II
    class Solution {
        fun nextGreaterElements(nums: IntArray): IntArray {
            val n = nums.size
            val ret = IntArray(n) { -1 }

            val stack = LinkedList<Int>()
            for (i in 0 until n * 2 - 1) {
                while (stack.isNotEmpty() && nums[stack.peek()] < nums[i % n]) {
                    ret[stack.pop()] = nums[i % n]
                }
                stack.push(i % n)
            }

            return ret
        }
    }
}