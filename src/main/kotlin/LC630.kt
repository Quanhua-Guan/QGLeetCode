import java.util.*

class LC630 {
    /// 630. 课程表 III
    class Solution {
        fun scheduleCourse(courses: Array<IntArray>): Int {
            courses.sortBy { it.last() } // 根据结束时间升序排序
            val queue = PriorityQueue<Int> { a, b -> b - a } // 记录课程时长的大顶堆
            var sum = 0
            for (course in courses) {
                val d = course[0]
                val e = course[1]
                sum += d
                queue.add(d)
                if (sum > e) {
                    sum -= queue.poll()
                }
            }
            return queue.size
        }
    }
}