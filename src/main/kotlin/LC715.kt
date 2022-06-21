import java.util.*

class LC715 {
    class RangeModule_1() {
        // 保证每个区间不互相重合且递增排序
        val treeMap = TreeMap<Int, Int>()

        fun addRange(left: Int, right: Int) {
            var left = left
            var right = right

            var entry = treeMap.higherEntry(left)
            if (entry == null) {
                val start = treeMap.lastEntry()
                if (start == null) {
                    treeMap[left] = right
                } else {
                    // start.key <= left < right
                    if (start.value < left) {
                        treeMap[left] = right
                    } else if (start.value < right) {
                        // start.key <= left <= start.value < right
                        treeMap[start.key] = right
                    } else {
                        // start.key <= left < right <= start.value
                        // do nothing
                    }
                }
            } else {
                val start = treeMap.lowerEntry(entry.key)
                // start.key <= left < right
                // left < entry.key
                if (start == null) {
                    // do nothing here
                } else {
                    if (start.value < left) {
                        // do nothing here
                    } else if (start.value < right) {
                        // start.key <= left <= start.value < right
                        left = start.key
                        treeMap.remove(start.key)
                    } else { // start.value >= right
                        // start.key <= left < right <= start.value
                        return
                    }
                }
                while (entry != null && entry.key <= right) {
                    right = maxOf(right, entry.value)
                    treeMap.remove(entry.key)
                    entry = treeMap.higherEntry(entry.key)
                }
                treeMap[left] = right
            }
        }

        fun queryRange(left: Int, right: Int): Boolean {
            val entry = treeMap.higherEntry(left)
            val start = if (entry == null) treeMap.lastEntry() else treeMap.lowerEntry(entry.key)
            return start != null && start.value >= right
        }


        fun removeRange(left: Int, right: Int) {
            var entry = treeMap.higherEntry(left)
            if (entry == null) {
                val start = treeMap.lastEntry()
                if (start == null) {
                    return
                } else {
                    // start != null
                    // start.key <= left < right
                    if (start.value <= left) {
                        // do nothing here
                    } else if (start.value <= right) {
                        // start.key <= left < start.value <= right
                        treeMap[start.key] = left
                    } else {
                        // start.value <= left < right < start.value
                        treeMap[start.key] = left
                        treeMap[right] = start.value
                    }
                }
            } else {
                // entry != null
                val start = treeMap.lowerEntry(entry.key)
                if (start == null) {
                    // do nothing
                } else {
                    // start != null
                    // start.key <= left < right
                    if (start.value <= left) {
                        // do nothing
                    } else if (start.value <= right) {
                        // start.value <= left < start.value <= right
                        treeMap[start.key] = left
                    } else { // right < start.value
                        treeMap[start.key] = left
                        treeMap[right] = start.value
                    }
                }

                // start.key <= left < entry.key
                while (entry != null && entry.key < right) { // right < entry.value or right >= entry.value
                    treeMap.remove(entry.key)
                    if (right < entry.value) {
                        treeMap[right] = entry.value
                    }
                    entry = treeMap.higherEntry(entry.key)
                }
            }
        }
    }

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

    class RangeModule_20220621() {

        val tm = TreeMap<Int, Int>()

        fun addRange(left: Int, right: Int) {
            var left = left
            var right = right
            var entry = tm.higherEntry(left)
            if (entry == null) {
                val start = tm.lastEntry()
                if (start == null) {
                    tm[left] = right
                } else {
                    if (start.value < left) {
                        tm[left] = right
                    } else if (start.value < right) {
                        tm[start.key] = right
                    } else { // start.key <= left < right <= start.value
                        // do nothing
                    }
                }
            } else {
                // entry != null
                val start = tm.lowerEntry(entry.key)
                if (start != null) {
                    // start.key <= left < right
                    if (start.value < left) {

                    } else if (start.value <= right) {
                        tm.remove(start.key)
                        left = start.key
                    } else { // start.key <= left < right <= start.value
                        return
                    }
                }
                while (entry != null && entry.key <= right) {
                    right = maxOf(right, entry.value)
                    tm.remove(entry.key)
                    entry = tm.higherEntry(entry.key)
                }
                tm[left] = right
            }
        }

        fun queryRange(left: Int, right: Int): Boolean {
            val entry = tm.higherEntry(left)
            val start = if (entry == null) tm.lastEntry() else tm.lowerEntry(entry.key)
            return start != null && start.value >= right
        }

        fun removeRange(left: Int, right: Int) {
            var entry = tm.higherEntry(left)
            if (entry == null) {
                val start = tm.lastEntry()
                if (start == null) {
                    return
                } else {
                    // start.key <= left < right
                    if (start.value < left) {
                        return
                    } else if (start.value <= right) {
                        tm[start.key] = left
                    } else {
                        tm[start.key] = left
                        tm[right] = start.value
                    }
                }
            } else {
                // entry != null
                val start = tm.lowerEntry(entry.key)
                if (start != null) {
                    // start.key <= left < right
                    if (start.value < left) {

                    } else if (start.value <= right) {
                        tm[start.key] = left
                    } else { // start.key <= left < right < start.value
                        tm[start.key] = left
                        tm[right] = start.value
                    }
                }
                // left < entry.key < entry.value
                while (entry != null && entry.key < right) {
                    tm.remove(entry.key)
                    if (entry.value > right) {
                        tm[right] = entry.value
                    }
                    entry = tm.higherEntry(entry.key)
                }
            }
        }

    }
}