import com.cqmh.qgleetcode.*

fun main() {
    while (true) {
        try {
            println(
                    Solution1229().minAvailableDuration(
                            arrayOf(intArrayOf(216397070, 363167701), intArrayOf(98730764, 158208909), intArrayOf(441003187, 466254040), intArrayOf(558239978, 678368334), intArrayOf(683942980, 717766451)),
                            arrayOf(intArrayOf(50490609, 222653186), intArrayOf(512711631, 670791418), intArrayOf(730229023, 802410205), intArrayOf(812553104, 891266775), intArrayOf(230032010, 399152578)),
                            456085
                    )
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}