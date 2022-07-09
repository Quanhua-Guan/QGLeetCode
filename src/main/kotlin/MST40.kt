import java.util.*

class MST40 {
    /// 面试题40. 最小的k个数
    class Solution_sort {
        fun getLeastNumbers(arr: IntArray, k: Int): IntArray {
            arr.sort()
            return arr.copyOfRange(0, k)
        }
    }

    class Solution_heap {
        fun getLeastNumbers(arr: IntArray, k: Int): IntArray {
            if (k == 0) return intArrayOf()

            val mink = PriorityQueue<Int> { a, b -> b - a }
            arr.forEach {
                if (mink.size == k) {
                    val top = mink.peek()
                    if (top > it) {
                        mink.poll()
                        mink.add(it)
                    }
                } else {
                    mink.add(it)
                }
            }
            return mink.toIntArray()
        }
    }

    class Solution_heap2 {
        fun getLeastNumbers(arr: IntArray, k: Int): IntArray {
            val mink = PriorityQueue<Int>()
            arr.forEach {
                mink.offer(it)
            }
            var result = IntArray(k)
            for (i in 0 until k) {
                result[i] = mink.poll()
            }
            return result
        }
    }
}