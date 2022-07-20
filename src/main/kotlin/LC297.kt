import java.util.*

class LC297 {
    /**
     * Definition for a binary tree node.
     */
    class TreeNode(var `val`: Int) {
        var left: TreeNode? = null
        var right: TreeNode? = null
    }

    class Codec() {
        // Encodes a URL to a shortened URL.
        fun serialize(root: TreeNode?): String {
            if (root == null) return ""

            val data = StringBuilder()
            val queue = LinkedList<TreeNode?>()
            queue.offer(root)
            while (queue.isNotEmpty()) {
                val node = queue.poll()
                if (node != null) {
                    data.append(node.`val`.toString() + ',')
                    queue.offer(node.left)
                    queue.offer(node.right)
                } else {
                    data.append('x'.toString() + ',')
                }
            }
            data.deleteCharAt(data.length - 1)
            return data.toString()
        }

        // Decodes your encoded data to tree.
        fun deserialize(data: String): TreeNode? {
            if (data.isEmpty()) return null
            val datas = data.split(',')

            var i = 0
            val root = TreeNode(datas[i].toInt())

            val queue = LinkedList<TreeNode>()
            queue.offer(root)
            while (queue.isNotEmpty()) {
                val node = queue.pollFirst()
                i++
                if (datas[i] != "x") {
                    node.left = TreeNode(datas[i].toInt())
                    queue.offer(node.left)
                }
                i++
                if (datas[i] != "x") {
                    node.right = TreeNode(datas[i].toInt())
                    queue.offer(node.right)
                }
            }

            return root
        }
    }

    /**
     * Your Codec object will be instantiated and called as such:
     * var ser = Codec()
     * var deser = Codec()
     * var data = ser.serialize(longUrl)
     * var ans = deser.deserialize(data)
     */
}