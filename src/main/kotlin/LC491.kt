class LC491 {
    /// 491. 递增子序列
    class Solution {
        fun findSubsequences(nums: IntArray): List<List<Int>> {
            val n = nums.size
            var res = mutableSetOf<List<Int>>()
            var choosenMax = Math.pow(2.0, n.toDouble()).toInt() - 1
            for (choosen in 1..choosenMax) {
                val cur = mutableListOf<Int>()
                for (bit in 0 until n) {
                    if ((1 shl bit) and choosen != 0) {
                        val num = nums[bit]
                        if (cur.isEmpty() || num >= cur.last()!!) {
                            cur.add(num)
                        } else {
                            cur.clear()
                            break
                        }
                    }
                }
                if (cur.size >= 2) {
                    res.add(cur)
                }
            }
            return res.toList()
        }
    }

    class SolutionDFS {
        fun findSubsequences(nums: IntArray): List<List<Int>> {
            val n = nums.size
            var res = mutableSetOf<List<Int>>()

            fun dfs(index: Int, cur: MutableList<Int>) {
                if (index == n) {
                    if (cur.size >= 2) {
                        res.add(cur.toList())
                    }
                    return
                }

                val num = nums[index]
                if (cur.isEmpty() || num >= cur.last()!!) {
                    cur.add(num)
                    dfs(index + 1, cur)
                    cur.removeAt(cur.size - 1)
                }

                dfs(index + 1, cur)
            }

            dfs(0, mutableListOf<Int>())
            return res.toList()
        }
    }

    class SolutionDFS_OP {

        fun findSubsequences1(nums: IntArray): List<List<Int>> {
            val res = mutableListOf<List<Int>>()
            val path = mutableListOf<Int>()

            fun backTracking(start: Int) {
                val used = IntArray(201)
                if (path.size >= 2) res.add(path.toList())
                for (i in start..nums.lastIndex) {
                    if (path.isNotEmpty() && path.last() > nums[i] || used[nums[i] + 100] == 1) {
                        continue
                    } else {
                        used[nums[i] + 100] = 1
                        path.add(nums[i])
                        backTracking(i + 1)
                        path.removeAt(path.lastIndex)
                    }
                }
            }
            backTracking(0)
            return res
        }

        fun findSubsequences(nums: IntArray): List<List<Int>> {
            val result = mutableListOf<List<Int>>()
            val path = mutableListOf<Int>()

            fun backTracking(start: Int) {
                val used = BooleanArray(201)

                if (path.size >= 2) result.add(path.toList())

                for (i in start..nums.lastIndex) {
                    val num = nums[i]
                    if (path.isNotEmpty() && num < path.last()!! || used[num + 100]) {
                        continue
                    } else {
                        used[num + 100] = true
                        path.add(num)
                        backTracking(i + 1)
                        path.removeAt(path.lastIndex)
                    }
                }
            }

            backTracking(0)
            return result
        }
    }
}