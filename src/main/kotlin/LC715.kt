import java.util.*

class LC715 {
    /// 715. Range 模块
    class RangeModule() {

        val intervals = TreeMap<Int, Int>() // 存储有序不想交的区间

        fun addRange(left: Int, right: Int) {
            var left = left
            var right = right
            var entry = intervals.higherEntry(left)

            if (entry != intervals.firstEntry()) {
                val start = if (entry != null) intervals.lowerEntry(entry.key) else intervals.lastEntry()
                if (start != null && start.value >= right) {
                    return // [left, right) 已经属于管理区间的子区间, 不需要更多操作
                }
                if (start != null && start.value >= left) {
                    left = start.key // 区间合并
                    intervals.remove(start.key) // 删除原区间
                }
            }

            while (entry != null && entry.key <= right) {
                right = maxOf(right, entry.value)
                intervals.remove(entry.key)
                entry = intervals.higherEntry(entry.key)
            }

            intervals[left] = right
        }

        fun queryRange(left: Int, right: Int): Boolean {
            var entry = intervals.higherEntry(left)
            if (entry == intervals.firstEntry()) {
                return false
            }
            entry = if (entry != null) intervals.lowerEntry(entry.key) else intervals.lastEntry()
            return entry != null && right <= entry.value
        }

        fun removeRange(left: Int, right: Int) {
            var entry = intervals.higherEntry(left)
            if (entry != intervals.firstEntry()) {
                val start = if (entry != null) intervals.lowerEntry(entry.key) else intervals.lastEntry()

                // start.key <= left
                if (start != null && start.value >= right) {
                    val ri = start.value
                    if (start.key == left) {
                        intervals.remove(start.key)
                    } else { // start.key < left
                        intervals[start.key] = left // 更新
                    }
                    if (right != ri) {
                        intervals[right] = ri
                    }
                    return
                } else if (start != null && start.value > left) {
                    intervals[start.key] = left
                }
            }
            while (entry != null && entry.key < right) {
                if (entry.value <= right) {
                    // [entry.key, entry.value) 被 [left, right) 所包含, 直接删除
                    intervals.remove(entry.key)
                    entry = intervals.higherEntry(entry.key)
                } else { // entry.value > right
                    intervals[right] = entry.value
                    intervals.remove(entry.key)
                    break // 根据区间有序, 下一个entry不需要处理, 直接break
                }
            }
        }

    }

    /**
     * Your RangeModule object will be instantiated and called as such:
     * var obj = RangeModule()
     * obj.addRange(left,right)
     * var param_2 = obj.queryRange(left,right)
     * obj.removeRange(left,right)
     */
}