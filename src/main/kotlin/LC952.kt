import java.util.*
import kotlin.math.sqrt

class LC952 {
    /// 952. 按公因数计算最大组件大小
    class Solution {
        fun largestComponentSize(nums: IntArray): Int {
            val m = Collections.max(nums.map { it })
            val uf = UnionFind(m + 1)
            for (num in nums) {
                for (i in 2..Math.sqrt(num.toDouble()).toInt()) {
                    if (num % i == 0) {
                        uf.union(num, i)
                        uf.union(num, num / i)
                    }
                }
            }

            val counts = IntArray(m + 1)
            var ans = 0
            for (num in nums) {
                val root = uf.find(num)
                counts[root]++
                ans = maxOf(ans, counts[root])
            }

            return ans
        }

        class UnionFind(val n: Int) {
            var parent: IntArray = IntArray(n)
            var rank: IntArray = IntArray(n)
            init {
                for (i in 0 until n) {
                    parent[i] = i
                }
            }

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

    class Solution_timeout {

        /// 求解质因数列表集合
        fun primeFactors(n: Int): Set<Int> {
            assert(n > 1)
            val res = mutableSetOf<Int>()
            var num = n
            for (i in 2..sqrt(n.toDouble()).toInt()) {
                while (num % i == 0) {
                    res.add(i)
                    num /= i
                }
            }
            return res;
        }

        class Component() {
            var primeFactors = mutableSetOf<Int>()
            var nums = mutableSetOf<Int>()
        }

        fun largestComponentSize(nums: IntArray): Int {
            // 为每个组件构造一个公因数集合
            // 每次取一个数, 做因数分解, 每个因数
            //  - 判断是否在已存在的一个公因数集合

            val components = mutableListOf<Component>()
            var maxCount = Int.MIN_VALUE

            for (num in nums) {
                val indices = mutableListOf<Int>()
                val primeFactors = primeFactors(num)

                for (i in components.indices) {
                    val component = components[i]
                    for (factor in primeFactors) {
                        if (component.primeFactors.contains(factor)) {
                            indices.add(i)
                            break
                        }
                    }
                }

                val newComponent = Component()
                newComponent.primeFactors.addAll(primeFactors)
                newComponent.nums.add(num)
                components.add(newComponent)

                for (i in indices.reversed()) {
                    val component = components.removeAt(i)
                    newComponent.primeFactors.addAll(component.primeFactors)
                    newComponent.nums.addAll(component.nums)
                }

                if (newComponent.nums.size > maxCount) {
                    maxCount = newComponent.nums.size
                }
            }

            return maxCount
        }
    }
}