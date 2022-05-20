import com.cqmh.qgleetcode.*

fun main() {
    while (true) {
        try {
            val mat = arrayOf(
                intArrayOf(1,0,1,1,0,0,1,0,0,1),
                intArrayOf(0,1,1,0,1,0,1,0,1,1),
                intArrayOf(0,0,1,0,1,0,0,1,0,0),
                intArrayOf(1,0,1,0,1,1,1,1,1,1),
                intArrayOf(0,1,0,1,1,0,0,0,0,1),
                intArrayOf(0,0,1,0,1,1,1,0,1,0),
                intArrayOf(0,1,0,1,0,1,0,0,1,1),
                intArrayOf(1,0,0,0,1,1,1,1,0,1),
                intArrayOf(1,1,1,1,1,1,1,0,1,0),
                intArrayOf(1,1,1,1,0,1,0,0,1,1)
            )
            Solution542().updateMatrix(mat)
            println(mat)
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}