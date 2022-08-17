class LC1302 {
    /// 1302. 层数最深叶子节点的和
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
        fun deepestLeavesSum(root: TreeNode?): Int {
            var sum = 0
            var maxDepth = 0

            fun dfs(node: TreeNode?, depth: Int) {
                if (node == null) return

                if (node.left != null) {
                    dfs(node.left!!, depth + 1)
                }
                if (node.right != null) {
                    dfs(node.right!!, depth + 1)
                }

                if (node.left == null && node.right == null) {
                    if (maxDepth == depth) {
                        sum += node.`val`
                    } else if (maxDepth < depth) {
                        sum = node.`val`
                        maxDepth = depth
                    } else if (maxDepth > depth) {
                        // do nothing
                    }
                }
            }

            dfs(root, 0)
            return sum
        }
    }
}