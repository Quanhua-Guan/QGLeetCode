import java.util.*


fun main() {
    while (true) {
        try {
            println(
                LC749.Solution().containVirus(
                    arrayOf(
                        intArrayOf(0, 1, 0, 0, 0, 0, 0, 1),
                        intArrayOf(0, 1, 0, 0, 0, 0, 0, 1),
                        intArrayOf(0, 0, 0, 0, 0, 0, 0, 1),
                        intArrayOf(0, 0, 0, 0, 0, 0, 0, 0)
                    )
                )
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}


class LC749 {
    class Solution {
        var rowCount = 0
        var colCount = 0
        var visit = Array(1) { BooleanArray(1) }

        fun containVirus(isInfected: Array<IntArray>): Int {
            rowCount = isInfected.size
            colCount = isInfected[0].size

            var count = 0
            while (true) {
                val cnt = getWallCount(isInfected)
                if (cnt == 0) break
                count += cnt
            }
            return count
        }

        fun getWallCount(m: Array<IntArray>): Int {
            var maxWillInfectedCount = 0
            var maxWallCount = 0
            visit = Array(rowCount) { BooleanArray(colCount) }

            // 找到所有被感染的区域，以及对应的将感染的区域
            val infected = mutableListOf<Set<Pair<Int, Int>>>()
            val willInfected = mutableListOf<Set<Pair<Int, Int>>>()

            for (r in 0 until rowCount) {
                for (c in 0 until colCount) {
                    if (m[r][c] == 1 && !visit[r][c]) {
                        val infectedSet = mutableSetOf<Pair<Int, Int>>()
                        val willInfectedSet = mutableSetOf<Pair<Int, Int>>()
                        val wallCount = search(r, c, m, infectedSet, willInfectedSet)
                        if (willInfectedSet.size > maxWillInfectedCount) {
                            maxWillInfectedCount = willInfectedSet.size
                            maxWallCount = wallCount
                        }
                        infected.add(infectedSet)
                        willInfected.add(willInfectedSet)
                    }
                }
            }

            for (i in willInfected.indices) {
                val wi = willInfected[i]
                if (wi.size == maxWillInfectedCount) {
                    // 被隔离的区域
                    val infe = infected[i]
                    for ((r, c) in infe) {
                        m[r][c] = -1
                    }
                } else {
                    for ((r, c) in wi) {
                        m[r][c] = 1 // 本次未被隔离的区域将往外扩散病毒
                    }
                }
            }

            return maxWallCount
        }

        // 返回防火墙个数
        fun search(
            r: Int,
            c: Int,
            m: Array<IntArray>,
            infected: MutableSet<Pair<Int, Int>>,
            willInfected: MutableSet<Pair<Int, Int>>
        ): Int {
            var wallCount = 0

            val queue = ArrayDeque<Pair<Int, Int>>()
            queue.add(Pair(r, c))
            infected.add(Pair(r, c))
            visit[r][c] = true

            val direction = listOf(Pair(0, 1), Pair(0, -1), Pair(1, 0), Pair(-1, 0))
            while (queue.isNotEmpty()) {
                val (r, c) = queue.pollFirst()
                for ((dr, dc) in direction) {
                    val nr = r + dr
                    val nc = c + dc
                    if (nr < 0 || nr >= rowCount || nc < 0 || nc >= colCount || visit[nr][nc]) continue

                    if (m[nr][nc] == 1) {
                        infected.add(Pair(nr, nc))
                        visit[nr][nc] = true
                        queue.offerLast(Pair(nr, nc))
                    } else if (m[nr][nc] == 0) {
                        willInfected.add(Pair(nr, nc))
                        wallCount++
                    }
                }
            }

            return wallCount
        }
    }
}