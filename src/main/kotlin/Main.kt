import com.cqmh.qgleetcode.*

fun main() {
    while (true) {
        try {
            Solution462().minMoves2_(intArrayOf(2, 1, 3))
            var nums = intArrayOf(1, 2, 3, 7, 7, 7, 7, 6)
            Solution31().nextPermutation(nums)
            println(nums.joinToString(","))
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}