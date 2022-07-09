import java.util.*

class JZO41 {
    class MedianFinder() {

        // leftMaxHeap           rightMinHeap
        // leftMaxHeap 放置 (n + 1) / 2 个数,  rightMinHeap 放置 n / 2 个数

        val leftMaxHeap = PriorityQueue<Int> { a, b -> b - a }
        val rightMinHeap = PriorityQueue<Int>()

        fun addNum(num: Int) {
            if (leftMaxHeap.isEmpty() && rightMinHeap.isEmpty()) {
                leftMaxHeap.offer(num)
            } else {
                val leftTop = if (leftMaxHeap.isEmpty()) Int.MIN_VALUE else leftMaxHeap.peek()
                if (leftTop <= num) {
                    rightMinHeap.offer(num)
                } else {
                    leftMaxHeap.offer(num)
                }
            }

            while (leftMaxHeap.size > rightMinHeap.size + 1) {
                rightMinHeap.offer(leftMaxHeap.poll())
            }
            while (leftMaxHeap.size < rightMinHeap.size) {
                leftMaxHeap.offer(rightMinHeap.poll())
            }
        }

        fun findMedian(): Double {
            val leftSize = leftMaxHeap.size
            val rightSize = rightMinHeap.size
            if (leftSize == rightSize + 1) {
                return leftMaxHeap.peek().toDouble()
            }
            if (leftSize == 0) return 0.0
            return (leftMaxHeap.peek() + rightMinHeap.peek()) / 2.0
        }

    }

    /**
     * Your MedianFinder object will be instantiated and called as such:
     * var obj = MedianFinder()
     * obj.addNum(num)
     * var param_2 = obj.findMedian()
     */
}