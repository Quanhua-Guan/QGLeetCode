fun main() {
    while (true) {
        try {
            println(
                LC1224.Solution().maxEqualFreq(intArrayOf(1, 1, 1, 2, 2, 2, 3, 3, 3))
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}

class LC1224 {
    /// 1224. 最大相等频率
    class Solution {
        fun maxEqualFreq(nums: IntArray): Int {
            var maxRes = 2
            val numCount = mutableMapOf<Int, Int>()
            for (i in nums.indices) {
                val num = nums[i]
                numCount[num] = numCount.getOrDefault(num, 0) + 1

                val totalCount = i + 1
                val diffNumCount = numCount.size
                if (totalCount == diffNumCount || diffNumCount == 1) {
                    maxRes = totalCount
                } else {
                    if ((totalCount - 1) % diffNumCount == 0) {
                        val k = (totalCount - 1) / diffNumCount
                        var other = -1
                        var breaked = false
                        for (count in numCount.values) {
                            if (count == k) {
                            } else if (other == -1) {
                                other = count
                            } else if (count == other) {
                            } else {
                                breaked = true
                                break
                            }
                        }
                        if (!breaked && other == k + 1) {
                            maxRes = totalCount
                        }
                    }
                    if ((totalCount - 1) % (diffNumCount - 1) == 0) {
                        val k = (totalCount - 1) / (diffNumCount - 1)
                        var other = -1
                        var breaked = false
                        for (count in numCount.values) {
                            if (count == k) {
                            } else if (other == -1) {
                                other = count
                            } else if (count == other) {
                            } else {
                                breaked = true
                                break
                            }
                        }
                        if (!breaked && other == 1) {
                            maxRes = totalCount
                        }
                    }
                }
            }

            return maxRes
        }
    }

    class Solution_op {
        fun maxEqualFreq(nums: IntArray): Int {
            var maxRes = 2
            val numCount = mutableMapOf<Int, Int>()
            for (i in nums.indices) {
                val num = nums[i]
                numCount[num] = numCount.getOrDefault(num, 0) + 1

                val totalCount = i + 1
                val diffNumCount = numCount.size
                if (totalCount == diffNumCount || diffNumCount == 1) {
                    maxRes = totalCount
                } else {
                    if ((totalCount - 1) % diffNumCount == 0 || (totalCount - 1) % (diffNumCount - 1) == 0) {
                        var count1 = -1
                        var count1count = 0
                        var count2 = -1
                        var count2count = 0
                        var breaked = false
                        for (count in numCount.values) {
                            if (count1 == -1) {
                                count1 = count
                                count1count = 1
                            } else if (count == count1) {
                                ++count1count
                            } else if (count2 == -1) {
                                count2 = count
                                count2count = 1
                            } else if (count2 == count) {
                                ++count2count
                            } else {
                                breaked = true
                                break
                            }
                        }
                        if (!breaked && (
                                    (count1 == 1 && count1count == 1) ||
                                            (count2 == 1 && count2count == 1) ||
                                            (count1 == count2 + 1 && count1count == 1) ||
                                            (count2 == count1 + 1 && count2count == 1)
                                    )
                        ) {
                            maxRes = totalCount
                        }
                    }
                }
            }

            return maxRes
        }
    }

    class Solution_op_1 {
        fun maxEqualFreq(nums: IntArray): Int {
            var countNumMax = 0
            var res = 0

            // count[x]代表数字x出现次数
            val count = mutableMapOf<Int, Int>()
            // freq[count]代表出现次数为count次的数字的类数
            val freq = mutableMapOf<Int, Int>()

            for (i in nums.indices) {
                val num = nums[i]

                var countNum = count.getOrDefault(num, 0)
                if (countNum > 0) {
                    freq[countNum] = freq[countNum]!! - 1
                }

                countNum = count.getOrDefault(num, 0) + 1
                count[num] = countNum
                freq[countNum] = freq.getOrDefault(countNum, 0) + 1

                countNumMax = maxOf(countNumMax, countNum)

                if (countNumMax == 1 ||
                    (freq[countNumMax] == 1 && freq[countNumMax - 1]!! * (countNumMax - 1) + countNumMax == i + 1) ||
                    (freq[1] == 1 && freq[countNumMax]!! * countNumMax + 1 == i + 1)
                ) {
                    res = i + 1
                }
            }

            return res
        }

        fun maxEqualFreq_op(nums: IntArray): Int {
            var countNumMax = 0
            var res = 0

            // count[x]代表数字x出现次数
            val count = mutableMapOf<Int, Int>()
            // freq[count]代表出现次数为count次的数字的类数
            val freq = mutableMapOf<Int, Int>()

            for (i in nums.indices) {
                val num = nums[i]

                var countNum = count.getOrDefault(num, 0)
                if (countNum > 0) {
                    freq[countNum] = freq[countNum]!! - 1
                }

                countNum = count.getOrDefault(num, 0) + 1
                count[num] = countNum
                freq[countNum] = freq.getOrDefault(countNum, 0) + 1

                countNumMax = maxOf(countNumMax, countNum)

                if (countNumMax == 1 ||
                    (freq[countNumMax] == 1 && freq[countNumMax - 1]!! * (countNumMax - 1) + countNumMax == i + 1) ||
                    (freq[1] == 1 && freq[countNumMax]!! * countNumMax + 1 == i + 1)
                ) {
                    res = i + 1
                }
            }

            return res
        }
    }
}