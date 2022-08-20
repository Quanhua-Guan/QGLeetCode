import java.util.*

class LC654 {
    /// 654. 最大二叉树
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

    class Solution {
        fun constructMaximumBinaryTree(nums: IntArray): TreeNode? {
            val stack = LinkedList<TreeNode>()
            val n = nums.size
            var node: TreeNode? = null
            for (i in 0 until n + 1) {
                val num = if (i == n) 1001 else nums[i]
                node = TreeNode(num)
                if (stack.isEmpty() || stack.peek()!!.`val` > num) {
                    stack.push(node)
                } else { // stack.peek().`val` < num
                    var prevNode: TreeNode? = null
                    while (stack.isNotEmpty() && stack.peek()!!.`val` < num) {
                        val popped = stack.pop()
                        popped.right = prevNode
                        prevNode = popped
                    }
                    node.left = prevNode
                    stack.push(node)
                }
            }
            return node?.left
        }

        fun constructMaximumBinaryTree_1(nums: IntArray): TreeNode? {

            fun create(start: Int, end: Int): TreeNode? {
                if (start > end) return null
                if (start == end) return TreeNode(nums[start])

                var maxIndex = start
                for (i in (start + 1)..end) {
                    if (nums[i] > nums[maxIndex]) {
                        maxIndex = i
                    }
                }

                val root = TreeNode(nums[maxIndex])
                root.left = create(start, maxIndex - 1)
                root.right = create(maxIndex + 1, end)

                return root
            }

            return create(0, nums.size - 1)
        }
    }
}