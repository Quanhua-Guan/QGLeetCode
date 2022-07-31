import java.util.*

class LC1161 {
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

    /**
     * Example:
     * var ti = TreeNode(5)
     * var v = ti.`val`
     * Definition for a binary tree node.
     * class TreeNode(var `val`: Int) {
     *     var left: TreeNode? = null
     *     var right: TreeNode? = null
     * }
     */
    class Solution {
        fun maxLevelSum(root: TreeNode?): Int {
            if (root == null) return 0 // 空树特殊情况处理

            // 二叉树 每一层的所有节点值 的 加和列表
            val nodeSums = mutableListOf<Int>()

            // 执行加和
            fun addNodeSum(level: Int, value: Int) {
                // 准备号 nodeSums 加和统计列表, 避免下标越界问题.
                while (level >= nodeSums.size) {
                    nodeSums.add(0)
                }
                nodeSums[level] = nodeSums[level] + value;
            }

            val queue = LinkedList<Pair<TreeNode, Int>>()
            queue.offer(Pair(root, 0))

            // 层序遍历
            while (queue.isNotEmpty()) {
                val (node, level) = queue.poll()
                addNodeSum(level, node.`val`)

                if (node.left != null) {
                    queue.offer(Pair(node.left!!, level + 1))
                }
                if (node.right != null) {
                    queue.offer(Pair(node.right!!, level + 1))
                }
            }

            // 先将最后一个加和值设为初始值
            var maxSum = nodeSums.last()
            var maxLevel = nodeSums.size - 1

            for (i in nodeSums.size - 2 downTo 0) {
                val sum = nodeSums[i]
                if (sum >= maxSum) {
                    maxSum = sum
                    maxLevel = i
                }
            }

            return maxLevel + 1 // 由于我们的level是从0开始的, 根据题意, 需要 +1
        }
    }
}