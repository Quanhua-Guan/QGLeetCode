class JZO34 {
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
        fun pathSum(root: TreeNode?, target: Int): List<List<Int>> {
            fun dfs(root: TreeNode?, target: Int): MutableList<MutableList<Int>> {
                if (root == null) return mutableListOf()

                if (root.left == null && root.right == null) {
                    if (root.`val` == target) {
                        return mutableListOf(mutableListOf(root.`val`))
                    }
                    return mutableListOf()
                } else {
                    val results = mutableListOf<MutableList<Int>>()
                    if (root.left != null) {
                        val list = dfs(root.left, target - root.`val`)
                        if (list.isNotEmpty()) {
                            list.forEach {
                                it.add(0, root.`val`)
                            }
                            results.addAll(list)
                        }
                    }
                    if (root.right != null) {
                        val list = dfs(root.right, target - root.`val`)
                        if (list.isNotEmpty()) {
                            list.forEach {
                                it.add(0, root.`val`)
                            }
                            results.addAll(list)
                        }
                    }
                    return results
                }
            }
            return dfs(root, target)
        }
    }

    class Solution_BackTrace {
        fun pathSum(root: TreeNode?, target: Int): List<List<Int>> {
            root ?: return mutableListOf()
            return pathSumBackTracing(root, target) ?: return mutableListOf()
        }

        fun pathSumBackTracing(root: TreeNode, target: Int): List<List<Int>>? {
            if (root.left == null && root.right == null) {
                return if (root.`val` == target) {
                    mutableListOf(mutableListOf(target))
                } else {
                    null
                }
            }

            val left = root.left?.run {
                pathSumBackTracing(this, target - root.`val`)?.let { list ->
                    list.onEach {
                        (it as? ArrayList<Int>)?.add(0, root.`val`)
                    }
                }
            }

            val right = root.right?.run {
                pathSumBackTracing(this, target - root.`val`)?.let { list ->
                    list.onEach {
                        (it as? ArrayList<Int>)?.add(0, root.`val`)
                    }
                }
            }

            return mutableListOf<List<Int>>().apply {
                left?.let {
                    addAll(it)
                }
                right?.let {
                    addAll(it)
                }
            }
        }
    }
}