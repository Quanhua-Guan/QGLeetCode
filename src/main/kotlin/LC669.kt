class LC669 {
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
        fun trimBST(root: TreeNode?, low: Int, high: Int): TreeNode? {
            val prehead = TreeNode(Int.MAX_VALUE)
            prehead.left = root

            fun trim(node: TreeNode?, parent: TreeNode) {
                if (node == null) return

                if (low <= node.`val` && node.`val` <= high) {
                    trim(node.left, node)
                    trim(node.right, node)
                } else if (node.`val` < low) {
                    if (parent.left == node) {
                        parent.left = node.right
                    } else {
                        parent.right = node.right
                    }
                    trim(node.right, parent)
                } else {
                    if (parent.left == node) {
                        parent.left = node.left
                    } else {
                        parent.right = node.left
                    }
                    trim(node.left, parent)
                }
            }

            trim(root, prehead)
            return prehead.left
        }
    }
}