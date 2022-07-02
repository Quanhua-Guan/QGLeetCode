import java.util.*

class LC841 {
    /// 871. 最低加油次数
    class Solution {
        fun minRefuelStops(target: Int, startFuel: Int, stations: Array<IntArray>): Int {
            val queue = PriorityQueue<Int> { f1, f2 -> f2 - f1 }
            var fuel = startFuel

            var count = 0
            for (station in stations) {
                val stationLocation = station[0]
                val stationFuel = station[1]

                while (fuel < stationLocation && queue.isNotEmpty()) {
                    fuel += queue.poll() // 加油一次 (贪心 - 每次加油总是选当前可选的可加油最多的那个加油站)
                    count += 1
                }

                if (fuel >= stationLocation) {
                    queue.add(stationFuel) // 将加油站添加到备选
                } else {
                    break
                }

                if (fuel >= target) break
            }

            while (fuel < target && queue.isNotEmpty()) {
                fuel += queue.poll()
                count++
            }

            if (fuel >= target) return count
            return -1
        }
    }
}