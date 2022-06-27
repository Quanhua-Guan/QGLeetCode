import java.util.*

class JZO32I {

    class TreeNode(var `val`: Int) {
        var left: TreeNode? = null
        var right: TreeNode? = null
    }

    class Solution {

        fun levelOrder(root: TreeNode?): IntArray {
            val result = mutableListOf<Int>()
            val q = LinkedList<TreeNode>()
            if (root != null) {
                q.offer(root)
            }
            while (q.isNotEmpty()) {
                val r = q.poll()
                result.add(r.`val`)

                if (r.left != null) {
                    q.offer(r.left!!)
                }
                if (r.right != null) {
                    q.offer(r.right!!)
                }
            }
            return result.toIntArray()
        }
    }
}