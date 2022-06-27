import java.util.*

class JZO32III {
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
        fun travel(root: TreeNode?, depth: Int, results: MutableList<MutableList<Int>>) {
            if (root == null) return

            assert(depth <= results.size)
            if (results.size == depth) {
                results.add(LinkedList<Int>())
            }
            if (depth and 1 == 1) {
                // 奇数层, 往前面加
                results[depth]!!.add(0, root.`val`)
            } else {
                // 偶数层, 往后面加
                results[depth]!!.add(root.`val`)
            }

            travel(root.left, depth + 1, results)
            travel(root.right, depth + 1, results)
        }

        fun levelOrder(root: TreeNode?): List<List<Int>> {
            val results = mutableListOf<MutableList<Int>>()
            travel(root, 0, results)
            return results
        }
    }
}