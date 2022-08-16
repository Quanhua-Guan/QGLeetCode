class LC1656 {
    /// 1656. 设计有序流
    class OrderedStream(val n: Int) {

        val data = Array<String?>(n + 1) { null }
        var ptr = 1

        fun insert(index: Int, value: String): List<String> {
            data[index] = value

            if (ptr < 1 || ptr > n || data[ptr] == null) {
                return emptyList()
            }

            val list = mutableListOf<String>()
            while (ptr <= n && data[ptr] != null) {
                list.add(data[ptr]!!)
                ++ptr
            }
            return list
        }

    }

    /**
     * Your OrderedStream object will be instantiated and called as such:
     * var obj = OrderedStream(n)
     * var param_1 = obj.insert(idKey,value)
     */
}