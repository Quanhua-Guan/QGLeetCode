import java.util.*

class LC768 {
    /// 768. 最多能完成排序的块 II
    class Solution {
        fun maxChunksToSorted_stack(nums: IntArray): Int {
            val stack = Stack<Int>()

            for (n in nums) {
                if (stack.isEmpty() || stack.peek() <= n) {
                    stack.push(n)
                } else {
                    val top = stack.pop()
                    while (stack.isNotEmpty() && stack.peek() > n) {
                        stack.pop()
                    }
                    stack.push(top)
                }
            }

            return stack.size
        }

        fun maxChunksToSorted_1(arr: IntArray): Int {
            var list = arr.mapIndexed { index, it -> Pair(it, index) }.sortedBy { it.first }

            // arr 重新编码, 元素取值范围 0 ~ n-1
            for (i in list.indices) {
                val pair = list[i]
                arr[pair.second] = i
            }

            val uf = UnionFind(list.size)
            for (i in list.indices) {
                val pair = list[i]
                val min = minOf(i, pair.second)
                val max = maxOf(i, pair.second)
                for (j in min until max) {
                    uf.union(arr[j], arr[max])
                }
            }

            val roots = mutableSetOf<Int>()
            for (i in 0 until uf.n) {
                roots.add(uf.find(i))
            }
            return roots.size
        }

        class UnionFind(val n: Int) {
            var parent = IntArray(n) { it }
            var rank = IntArray(n)

            fun union(x: Int, y: Int) {
                val rootx = find(x)
                val rooty = find(y)
                if (rootx != rooty) {
                    if (rank[rootx] > rank[rooty]) {
                        parent[rooty] = rootx
                    } else if (rank[rootx] < rank[rooty]) {
                        parent[rootx] = rooty
                    } else {
                        parent[rooty] = rootx
                        rank[rootx]++
                    }
                }
            }

            fun find(x: Int): Int {
                if (parent[x] != x) {
                    parent[x] = find(parent[x])
                }
                return parent[x]
            }
        }
    }
}