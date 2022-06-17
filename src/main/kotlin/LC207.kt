import java.util.*

fun main() {
    while (true) {
        try {
            println(
                LC207.Solution().canFinish(2, arrayOf(intArrayOf(1, 0)))
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}

class LC207 {
    // 207. 课程表
    class Solution {
        fun canFinish(numCourses: Int, prerequisites: Array<IntArray>): Boolean {
            val n = numCourses;
            val indegrees = Array(n) { mutableSetOf<Int>() }
            val outdegrees = Array(n) { mutableSetOf<Int>() }
            for (tofrom in prerequisites) {
                val from = tofrom[1]
                val to = tofrom[0]
                indegrees[to].add(from)
                outdegrees[from].add(to)
            }

            val nonzeroouts = mutableMapOf<Int, MutableSet<Int>>()
            val nonzeroins = mutableMapOf<Int, MutableSet<Int>>()
            val zeroins = LinkedList<Int>()
            for (i in 0 until n) {
                val indegree = indegrees[i]
                if (indegree.isEmpty()) {
                    zeroins.offer(i)
                } else {
                    nonzeroins[i] = indegree
                }

                val outdegree = outdegrees[i]
                if (outdegree.isNotEmpty()) {
                    nonzeroouts[i] = outdegree
                }
            }

            while (zeroins.isNotEmpty()) {
                val from = zeroins.poll()
                if (!nonzeroouts.containsKey(from)) continue

                val outdegree = nonzeroouts[from]!!
                for (to in outdegree) {
                    if (nonzeroins.containsKey(to)) {
                        nonzeroins[to]!!.remove(from)
                        if (nonzeroins[to]!!.isEmpty()) {
                            zeroins.add(to)
                            nonzeroins.remove(to)
                        }
                    }
                }
            }

            return nonzeroins.isEmpty()
        }
    }
}