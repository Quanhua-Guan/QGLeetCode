class LC655 {
    /// 655. 输出二叉树
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
        fun printTree(root: TreeNode?): List<List<String>> {
            var maxHeight = 0
            fun getHeight(root: TreeNode?, height: Int) {
                if (root == null) {
                    maxHeight = maxOf(maxHeight, height)
                    return
                }

                if (root.left != null) {
                    getHeight(root.left, height + 1)
                }
                if (root.right != null) {
                    getHeight(root.right, height + 1)
                }
            }
            getHeight(root, 0)

            val m = maxHeight + 1
            val n = Math.pow(2.0, maxHeight + 1.0).toInt() - 1
            val result: MutableList<MutableList<String>> = MutableList(m) {
                MutableList(n) {
                    ""
                }
            }

            fun travel(node: TreeNode?, r: Int, c: Int) {
                if (node == null) return

                result[r][c] = node.`val`.toString()

                val next = Math.pow(2.0, maxHeight - r - 1.0).toInt()
                travel(node.left, r + 1, c - next)
                travel(node.right, r + 1, c + next)
            }
            travel(root, 0, (n - 1) / 2)
            return result
        }
    }
}