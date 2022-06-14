import java.util.*

class LC384 {
    class Solution(val nums: IntArray) {

        val rand = Random()

        fun reset(): IntArray {
            return nums.clone()
        }

        fun shuffle(): IntArray {
            val shuffled = nums.clone()
            for (i in nums.size downTo 1) {
                val randomIndex = rand.nextInt(i)
                if (randomIndex != i) {
                    shuffled[randomIndex] = shuffled[i - 1].also { shuffled[i - 1] = shuffled[randomIndex] }
                }
            }
            return shuffled
        }

    }

    /**
     * Your Solution object will be instantiated and called as such:
     * var obj = Solution(nums)
     * var param_1 = obj.reset()
     * var param_2 = obj.shuffle()
     */
}