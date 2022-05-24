import com.cqmh.qgleetcode.*

fun main() {
    while (true) {
        try {
            println(
                Solution407().trapRainWater_BFS(
                    arrayOf(intArrayOf(5,8,7,7),intArrayOf(5,2,1,5),intArrayOf(7,1,7,1),intArrayOf(8,9,6,9),intArrayOf(9,8,9,9))
                )
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}