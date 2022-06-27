import java.util.*

class JZO32II {
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
            results[depth]!!.add(root.`val`)

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