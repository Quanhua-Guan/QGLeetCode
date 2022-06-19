fun main() {
    while (true) {
        try {
            val root = LC508.TreeNode(5)
            val left = LC508.TreeNode(2)
            val right = LC508.TreeNode(-3)
            root.left = left
            root.right = right
            println(
                LC508.Solution().findFrequentTreeSum(root)
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}

class LC508 {
    /**
     * Example:
     * var ti = TreeNode(5)
     * var v = ti.`val`
     * Definition for a binary tree node.
     */
    class TreeNode(var `val`: Int) {
        var left: TreeNode? = null
        var right: TreeNode? = null
    }

    /// 508. 出现次数最多的子树元素和
    class Solution {
        fun findFrequentTreeSum(root: TreeNode?): IntArray {
            // 遍历每一个子树, 记录每一个子树和,
            // 以子树和为 key 值, 记录和为该值的子树数

            val sumCount = mutableMapOf<Int, Int>()
            var maxCount = 0
            fun searchSubtreeSum(root: TreeNode?): Int {
                if (root == null) return 0

                // 左子树和
                var leftSum = searchSubtreeSum(root.left)
                var rightSum = searchSubtreeSum(root.right)
                return (root.`val` + leftSum + rightSum).also { sum: Int ->
                    val count = sumCount.getOrDefault(sum, 0) + 1
                    sumCount[sum] = count
                    if (maxCount < count) {
                        maxCount = count
                    }
                }
            }

            searchSubtreeSum(root)

            val results = mutableListOf<Int>()
            for ((k, v) in sumCount) {
                if (v == maxCount) {
                    results.add(k)
                }
            }
            return results.toIntArray()
        }
    }
}