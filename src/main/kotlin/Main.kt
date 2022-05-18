import com.cqmh.qgleetcode.*

fun main() {
    while (true) {
        try {
            var nums = intArrayOf(1,2,3,7,7,7,7,6)
            Solution31().nextPermutation(nums)
            println(nums.joinToString(","))
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}