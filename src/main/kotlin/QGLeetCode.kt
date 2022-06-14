package com.cqmh.qgleetcode

import java.lang.StringBuilder
import java.math.BigInteger
import java.util.*
import kotlin.collections.ArrayDeque
import kotlin.collections.ArrayList


/// 二分查找
class BinarySearch {
    companion object {
        /// 升序数组，查找某个数字下标
        fun indexOf(nums: IntArray, target: Int, from: Int? = null, to: Int? = null): Int {
            var l = from ?: 0
            var r = to ?: nums.size - 1

            while (l <= r) { // 包含相等，考虑 nums 仅包含 target 的情况
                var mid = (l + r) ushr 1
                if (nums[mid] > target) {
                    r = mid - 1
                } else if (nums[mid] < target) {
                    l = mid + 1
                } else {
                    return mid
                }
            }
            return -1
        }

        /// 升序数组，查找第一个大于或等于目标数的元素下标
        fun indexOfGreaterThanOrEqual(
            nums: IntArray,
            target: Int,
            from: Int? = null,
            to: Int? = null
        ): Int {
            var l = from ?: 0
            var r = to ?: nums.size - 1

            while (l <= r) {
                var mid = (l + r) ushr 1
                if (nums[mid] < target) {
                    l = mid + 1
                } else { // nums[mid] >= target
                    if (mid == 0 || nums[mid - 1] < target) {
                        return mid
                    } else {
                        r = mid - 1
                    }
                }
            }
            return -1
        }

        /// 升序数组，查找最后一个小于或等于目标数的元素下标
        fun indexOfLessThanOrEqual(
            nums: IntArray,
            target: Int,
            from: Int? = null,
            to: Int? = null
        ): Int {
            var l = from ?: 0
            var r = to ?: nums.size - 1

            while (l <= r) {
                var mid = (l + r) ushr 1
                if (nums[mid] > target) {
                    r = mid - 1
                } else { // nums[mid] <= target
                    if (mid == nums.size - 1 || nums[mid + 1] > target) {
                        return mid
                    } else {
                        l = mid + 1
                    }
                }
            }
            return -1
        }
    }
}

// 两数之和
class Solution1 {
    class IndexedNumber(val value: Int, val index: Int) {}

    fun twoSum3(nums: IntArray, target: Int): IntArray {
        var indexedNums = nums.mapIndexed({ index, value -> IndexedNumber(value, index) })
        indexedNums = indexedNums.sortedBy { it.value }
        var l = 0
        var r = indexedNums.size - 1

        while (l < r) {
            val left = indexedNums[l]
            val right = indexedNums[r]
            val sum = left.value + right.value
            if (sum == target) {
                return arrayOf(left.index, right.index).toIntArray()
            }
            if (sum < target) {
                l++
            } else {
                r--
            }
        }

        return intArrayOf()
    }

    fun twoSum1(nums: IntArray, target: Int): IntArray {
        // 存储数字和它的下标
        var numsMap = mutableMapOf<Int, Int>()

        for (i in nums.indices) {
            if (numsMap.containsKey(target - nums[i])) {
                return arrayOf(numsMap[target - nums[i]]!!, i).toIntArray()
            }
            numsMap[nums[i]] = i
        }

        return intArrayOf()
    }

    fun twoSum(nums: IntArray, target: Int): IntArray {
        var numMap = mutableMapOf<Int, Int>()
        var index = 0
        var result = intArrayOf(0, 0)
        for (num in nums) {
            if (numMap.containsKey(target - num)) {
                result[0] = numMap[target - num]!!
                result[1] = index
                break;
            } else {
                numMap[num] = index
            }
            index++
        }

        return result;
    }

    fun twoSum0(nums: IntArray, target: Int): IntArray {
        var numMap = mutableMapOf<Int, MutableList<Int>>()
        var index = 0
        for (num in nums) {
            if (!numMap.containsKey(num)) {
                var list = mutableListOf<Int>()
                list.add(index)
                numMap[num] = list
            } else {
                var list = numMap[num]!!
                list.add(index)
            }
            index++
        }

        val result = IntArray(2)
        for (num in nums) {
            val other = target - num
            if (numMap.containsKey(num) && numMap.containsKey(other)) {
                val numList = numMap[num]!!
                if (num == other) {
                    if (numList.size >= 2) {
                        result[0] = numList[0]
                        result[1] = numList[1]
                        break;
                    }
                } else {
                    val otherList = numMap[other]!!
                    result[0] = numList[0]
                    result[1] = otherList[0]
                }
            }
        }

        return result;
    }
}

class ListNode(var `val`: Int, var next: ListNode? = null) {}

// 2. 两数相加
class Solution2 {
    fun addTwoNumbers1(l1: ListNode?, l2: ListNode?): ListNode? {
        var l = l1
        var r = l2
        var carry = 0 // 进位
        var sumPrehead = ListNode(0)
        var sumNode = sumPrehead

        while (l != null || r != null) {
            var sum = (l?.`val` ?: 0) + (r?.`val` ?: 0) + carry
            carry = sum / 10
            sum %= 10

            sumNode.next = ListNode(sum)
            sumNode = sumNode.next!!

            l = l?.next
            r = r?.next
        }

        if (carry > 0) {
            sumNode.next = ListNode(carry)
        }

        return sumPrehead.next
    }

    fun addTwoNumbers(l1: ListNode?, l2: ListNode?): ListNode? {
        var l1 = l1
        var l2 = l2
        var carry = 0 // 进位
        var result = ListNode(-1)
        var current = result

        // 1, 2, 3
        // 2, 3,

        // 1, 2, 3
        // 1, 2, 3

        // 1, 2
        // 1, 2, 3

        while (l1 != null || l2 != null) {
            val val1 = l1?.`val` ?: 0
            var val2 = l2?.`val` ?: 0

            val sum = val1 + val2 + carry
            carry = sum / 10

            current.next = ListNode(sum % 10)
            current = current.next!!

            l1 = l1?.next
            l2 = l2?.next
        }

        if (carry > 0) {
            current.next = ListNode(carry)
        }

        return result.next
    }
}

class Solution9 {
    fun isPalindrome(x: Int): Boolean {
        if (x < 0) return false

        val nums = mutableListOf<Int>()
        var x = x
        while (x != 0) {
            nums.add(x % 10)
            x /= 10
        }

        var l = 0
        var r = nums.size - 1
        while (l < r) {
            if (nums[l] != nums[r]) return false
            l++
            r--
        }

        return true
    }

    fun isPalindrome1(x: Int): Boolean {
        if (x < 0) return false

        // 121 => 12 => 1 => 0
        // 1   => 12 => 121 =>x

        var xx = x
        var invertX = 0
        while (xx != 0) {
            invertX = invertX * 10 + (xx % 10)
            xx /= 10
        }

        return x == invertX
    }
}

class Solution3 {
    fun lengthOfLongestSubstring(s: String): Int {
        // 3个关键词
        // 最长, 记录 max
        // 无重复, 用字典记录是否已存在
        // 子串, 自测必然有 start 和 end 两个下标
        var charIndexMap = mutableMapOf<Char, Int>()
        var maxLength = 0
        var start = 0
        var current = 0

        while (current < s.length) {
            val char = s[current]
            if (charIndexMap.containsKey(char) && charIndexMap[char]!! >= start) {
                start = charIndexMap[char]!! + 1
            } else {
                maxLength = maxOf(maxLength, current - start + 1)
            }
            charIndexMap[char] = current
            current++
        }

        return maxLength
    }

    fun lengthOfLongestSubstring20220517(s: String): Int {
        // 无重复
        // 最长
        // 子串
        var start = 0
        var max = 0
        var mem = mutableMapOf<Char, Int>() // 存放出现过的字符所在的下标（如果字符重复，则只保持距离当前遍历到的下标最近的那个）

        var current = 0
        while (current < s.length) {
            val c = s[current]
            if (mem.containsKey(c) && mem[c]!! >= start) {
                start = mem[c]!! + 1
            }
            max = maxOf(max, current - start + 1)
            mem[c] = current

            current++
        }

        return max
    }
}

class Solution647 {
    // 回文子串数
    /**
    给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。
    回文字符串 是正着读和倒过来读一样的字符串。
    子字符串 是字符串中的由连续字符组成的一个序列。
    具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。
     */

    ///// 暴力解法
    fun countSubstrings(s: String): Int {
        var count = 0
        for (i in s.indices) {
            for (j in i + 1 until s.length) {
                val substring = s.substring(i, j)
                if (isPalindrome(substring)) {
                    count += 1
                }
            }
        }
        return count
    }

    fun isPalindrome(s: String): Boolean {
        var l = 0
        var r = s.length - 1
        while (l < r) {
            if (s[l] != s[r]) {
                return false
            }
            l++
            r--
        }
        return true
    }

    ///// 中心扩散法

    fun countSubstrings1(s: String): Int {
        var count = 0
        for (i in s.indices) {
            count += palindromeSubstringAtCenter(s, i)
        }
        return count
    }

    fun palindromeSubstringAtCenter(s: String, index: Int): Int {
        var count = 0

        // 以当前下标为中心的子串 (基数长度)
        var l = index
        var r = index
        while (l >= 0 && r < s.length && s[l] == s[r]) {
            count++
            l--
            r++
        }

        // 以当前下标为中左的子串 (偶数长度)
        l = index
        r = index + 1
        while (l >= 0 && r < s.length && s[l] == s[r]) {
            count++
            l--
            r++
        }

        return count
    }
}

/// 908. 最小差值 I
class Solution908 {
    // 思考: 读懂题意很重要, 是重要的第一步.
    // 计算两个数之间差值的最小值
    fun smallestRangeI(nums: IntArray, k: Int): Int {
        var min = Int.MAX_VALUE
        var max = Int.MIN_VALUE
        for (i in nums.indices) {
            min = minOf(min, nums[i])
            max = maxOf(max, nums[i])
        }

        if (max - min <= 2 * k) {
            return 0
        }
        return max - min - 2 * k
    }
}

/// 最长回文子串
class Solution5 {

    ///// 中心扩散法

    fun longestPalindrome(s: String): String {
        var longestLength = 0
        var center = -1
        for (i in s.indices) {
            val length = longestPalindromeLengthWithCenter(s, i)
            if (length > longestLength) {
                longestLength = length
                center = i
            }
        }

        val step = longestLength / 2
        if (longestLength % 2 == 1) {
            return s.substring(center - step, center + step + 1)
        }
        return s.substring(center - step + 1, center + step + 1)
    }

    fun longestPalindromeLengthWithCenter(s: String, center: Int): Int {
        if (s.isEmpty()) return 0

        var max = 0
        var l = center
        var r = center
        while (l >= 0 && r < s.length && s[l] == s[r]) {
            max = maxOf(max, r - l + 1)
            l--
            r++
        }

        l = center
        r = center + 1
        while (l >= 0 && r < s.length && s[l] == s[r]) {
            max = maxOf(max, r - l + 1)
            l--
            r++
        }

        return max
    }

    ///// 动态规划

    // dp[i][j] = s[i] == s[j] && ((j - 1) - (i + 1) < 1 || dp[i+1][j-1])
    // dp[i][j] = s[i] == s[j] && (j - i < 3 || dp[i+1][j-1])
    // i < j
    fun longestPalindrome_dp(s: String): String {
        if (s.isEmpty()) return ""

        val dp = Array(s.length) { BooleanArray(s.length) }

        var maxLength = 0
        var start = 0

        for (j in 1 until s.length) {
            for (i in 0 until j) {
                if (s[i] != s[j]) {
                    dp[i][j] = false
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true
                    } else {
                        dp[i][j] = dp[i + 1][j - 1]
                    }
                }

                if (dp[i][j] && j - i + 1 > maxLength) {
                    maxLength = j - i + 1
                    start = i
                }
            }
        }

        return s.substring(start, start + maxLength)
    }
}


/// 7. 整数翻转
/*
给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。

如果反转后整数超过 32 位的有符号整数的范围[−(2^31), (2^31) − 1] ，就返回 0。

假设环境不允许存储 64 位整数（有符号或无符号）。
* */
class Solution7_20220522 {
    fun reverse(xx: Int): Int {
        if (xx == Int.MIN_VALUE) return 0

        var x = if (xx < 0) -xx else xx
        var result = 0
        while (x > 0) {
            val n = x % 10
            if (result * 10 / 10 != result) {
                return 0
            }
            result *= 10
            if (result + n < result) {
                return 0
            }
            result += n

            x /= 10
        }
        return if (xx < 0) -result else result
    }
}

// 整数反转
class Solution7 {
    fun reverse(x: Int): Int {
        if (x == Int.MIN_VALUE) return 0

        val isNagtive = x < 0
        var x = if (isNagtive) -x else x
        var result = 0
        while (x != 0) {
            if ((result * 10 + (x % 10) - (x % 10)) / 10 != result) {
                return 0 // 注意溢出问题
            }
            result = result * 10 + (x % 10)
            x /= 10
        }

        return if (isNagtive) -result else result
    }
}

// 8. 字符串转换整数 (atoi)
class Solution8 {
    fun myAtoi(s: String): Int {
        var i = 0
        while (i < s.length && s[i] == ' ') {
            i++
        }

        var sign = 1
        if (i < s.length) {
            if (s[i] == '+') {
                i++
            } else if (s[i] == '-') {
                sign = -1
                i++
            }
        }

        var result = 0
        if (i < s.length && (s[i] in '0'..'9')) {
            result = (s[i] - '0') * sign
            i++
        }

        while (i < s.length && (s[i] in '0'..'9')) {
            val digit = (s[i] - '0') * sign
            if (result * 10 / 10 != result || (result < 0 && result * 10 + digit > 0) || (result > 0 && result * 10 + digit < 0)) {
                // 越界了
                return if (sign == 1) MAX_VALUE else MIN_VALUE
            }
            result = result * 10 + digit

            i++
        }

        return result
    }

    companion object {
        public const val MIN_VALUE: Int = -2147483648
        public const val MAX_VALUE: Int = 2147483647
    }
}

// Z 字形变换
class Solution6 {
    fun convert(s: String, numRows: Int): String {
        if (s.length <= 1 || numRows == 1) return s

        var charLists = List(numRows) { LinkedList<Char>() }

        // if numRows == 4
        // 0, 1, 2, 3, 2, 1
        var index = 0
        var ascend = true
        var i = 0
        while (i < s.length) {
            val char = s[i]
            charLists[index].add(char)

            if (ascend) {
                if (index == numRows - 1) {
                    ascend = false
                    index--
                } else {
                    index++
                }
            } else {
                if (index == 0) {
                    ascend = true
                    index++
                } else {
                    index--
                }
            }

            i++
        }

        var s = StringBuffer()
        for (charList in charLists) {
            charList.forEach { s.append(it) }
        }

        return s.toString()
    }
}

/// 4. 寻找两个正序数组的中位数
class Solution4 {
    fun findMedianSortedArrays(nums1: IntArray, nums2: IntArray): Double {
        val totalSize = nums1.size + nums2.size
        if (totalSize % 2 == 1) { // 总共 3 个数, 则 3 / 2 = 1, 应取第 3 / 2 + 1 个数
            return findNth(nums1, nums2, totalSize / 2 + 1)
        }
        // 总共 4 个数, 则 4 / 2 = 2, 应取第 2 和第 3 个数求平均值
        return (findNth(nums1, nums2, totalSize / 2) + findNth(nums1, nums2, totalSize / 2 + 1)) / 2
    }

    /// 查找两个有序数组的第 N 小的数
    fun findNth(nums1: IntArray, nums2: IntArray, n: Int): Double {
        var start1 = 0
        var start2 = 0
        /// 每次从 nums1 和 nums2 中, 以 start0 和 start1 为起点, 分别最多选出前 n/2 个数字构成 tmpNums1, tmpNums2, 则总共 选出了 n 个数字, 从 tmpNums1 和 tmpNums2 中取最大数字v1, v2(即最后一个数),
        /// 比较 v1, v2
        /// - 如果 v1 == v2, 返回 v1
        /// - 如果 v1 < v2, 则 v2 在所有数字中最高排名为 n, v2 在所有数字钟最高排名为 n - 1, 说明 tmpNums1 中的所有数字必然不可能为第 n 个数, 所以可以排除. 然后更新 start1, n.
        /// - 如果 v1 > v2, 同理, 可以将 tmpNums2 中的所有数字排除. 然后更新 start1, n.
        var n = n

        while (n > 1) {
            // 取值前先算下标, 不能越界
            // nums1, 从 start1 开始(包括 start1)依次选择最多 n/2 个数字 (最少可能选择 0 个数)
            val i1 = if (start1 >= nums1.size) start1 else minOf(start1 + n / 2 - 1, nums1.size - 1)
            // nums2, 同理
            val i2 = if (start2 >= nums2.size) start2 else minOf(start2 + n / 2 - 1, nums2.size - 1)

            // 取值
            val v1 = nums1.getOrElse(i1) { Int.MAX_VALUE }
            val v2 = nums2.getOrElse(i2) { Int.MAX_VALUE }

            // 判断
            if (v1 < v2) {
                n -= i1 - start1 + 1
                start1 = i1 + 1
            } else {
                n -= i2 - start2 + 1
                start2 = i2 + 1
            }
        }

        // 此时 n == 1
        val v1 = nums1.getOrElse(start1) { Int.MAX_VALUE }
        val v2 = nums2.getOrElse(start2) { Int.MAX_VALUE }
        return minOf(v1, v2).toDouble()
    }
}

class Solution4_20220521 {
    fun findMedianSortedArrays2(nums1: IntArray, nums2: IntArray): Double {
        var count = nums1.size + nums2.size
        val isEven = count % 2 == 0
        count = count / 2

        var l1 = 0
        var l2 = 0

        var tmp = 0
        while (l1 + l2 <= count) {
            if (l2 >= nums2.size || (l1 < nums1.size && nums1[l1] < nums2[l2])) {
                if (!isEven && l1 + l2 == count) tmp += nums1[l1]
                if (isEven && (l1 + l2 == count || l1 + l2 == count - 1)) tmp += nums1[l1]
                l1++
            } else {
                if (!isEven && l1 + l2 == count) tmp += nums2[l2]
                if (isEven && (l1 + l2 == count || l1 + l2 == count - 1)) tmp += nums2[l2]
                l2++
            }
        }

        return tmp / (if (isEven) 2.0 else 1.0)
    }

    fun findMedianSortedArrays1(nums1: IntArray, nums2: IntArray): Double {
        var count = nums1.size + nums2.size
        val isEven = count % 2 == 0
        count = count / 2

        var l1 = 0
        var l2 = 0

        var tmp = 0
        while (count > 0) {
            count--
            if (l2 >= nums2.size || (l1 < nums1.size && nums1[l1] < nums2[l2])) {
                tmp = nums1[l1]
                l1++
            } else { // l1 >= nums1.size || nums1[l1] > nums2[l2]
                tmp = nums2[l2]
                l2++
            }
        }

        if (isEven) {
            return (tmp + minOf(
                nums1.getOrElse(l1) { Int.MAX_VALUE },
                nums2.getOrElse(l2) { Int.MAX_VALUE }).toDouble()) / 2
        } else {
            return minOf(
                nums1.getOrElse(l1) { Int.MAX_VALUE },
                nums2.getOrElse(l2) { Int.MAX_VALUE }).toDouble()
        }
    }

    fun findMedianSortedArrays(nums1: IntArray, nums2: IntArray): Double {
        /// 找到第中位数，
        /// 如果只有1个中位数 m1，则返回 (m1, m1)
        /// 如果有左右2个中位数 m1、m2，则返回 (m1, m2)
        fun findMiddle(): Pair<Int, Int> {
            val n = nums1.size + nums2.size
            val hasTwoNumbers = n % 2 == 0
            var nth = (if (hasTwoNumbers) n / 2 else n / 2 + 1)

            assert(nth <= 0 || nth > nums1.size + nums2.size)

            var s1 = 0
            var s2 = 0
            while (nth > 1) {
                val i1 = maxOf(minOf(s1 + nth / 2 - 1, nums1.size - 1), s1)
                val i2 = maxOf(minOf(s2 + nth / 2 - 1, nums2.size - 1), s2)
                val v1 = nums1.getOrElse(i1) { Int.MAX_VALUE }
                val v2 = nums2.getOrElse(i2) { Int.MAX_VALUE }

                if (v1 < v2) {
                    nth -= (i1 - s1 + 1)
                    s1 = i1 + 1
                } else { // v1 > v2
                    nth -= (i2 - s2 + 1)
                    s2 = i2 + 1
                }
            }

            var v1 = nums1.getOrElse(s1) { Int.MAX_VALUE }
            var v2 = nums2.getOrElse(s2) { Int.MAX_VALUE }
            if (v1 > v2) {
                v1 = v2.also { v2 = v1 }
            }
            var v3 = minOf(nums1.getOrElse(s1 + 1) { Int.MAX_VALUE },
                nums2.getOrElse(s2 + 1) { Int.MAX_VALUE })
            if (v2 > v3) {
                v2 = v3
            }
            return Pair(v1, (if (hasTwoNumbers) v2 else v1))
        }

        val (m1, m2) = findMiddle()
        return (m1 + m2) / 2.0
    }
}

/// 392. 判断子序列
class Solution392 {
    // 判断字符串 s 是否是字符串 t 的子序列
    fun isSubsequence(s: String, t: String): Boolean {
        if (s.length > t.length) return false
        if (s.isEmpty()) return true

        var iS = 0
        var iT = 0
        while (iS < s.length && iT < t.length) {
            var valS = s[iS]
            var valT = valS - 1
            while (iT < t.length) {
                valT = t[iT]
                if (valS == valT) {
                    break
                }
                iT++
            }

            if (valS != valT) {
                return false
            }

            iS++
            iT++
        }

        return iS == s.length
    }
}

//  188. 买卖股票的最佳时机 IV
class Solution188 {
    fun maxProfit(k: Int, prices: IntArray): Int {
        if (prices.isEmpty()) return 0

        // dp[j][i] 在第 i 天最多操作 j 次的最大利润
        // k + 1 记录操作 0, 1, 2, ...k 次 (共 k+1种情况)
        // prices.size 天, 第 0, 1, 2, ...prices.size-1 天
        var dp = Array(k + 1) { IntArray(prices.size) }

        for (j in 1 until k + 1) { // 操作 j 次, j == 0时, 默认 dp[j][i] == 0, 所以从 j=1开始
            // 表示手上持有股票的前提下，i-1 天的总收益最大值
            // 第 0 天第一次操作后的收益（第一次操作一定是买入）
            var yestodayMaxWhenHold = -prices[0]
            for (i in 1 until prices.size) {
                // 第 i 天，第一天(i==0)只可能 1）什么都不做， 2）买入， 第二天开始出了以上两项， 增加操作 3）卖出
                //  所以第一天 dp[j][0] 最优质都为0， 所以 i 从 1 开始遍历（即从第二天开始）。
                dp[j][i] = maxOf(
                    //  第 i 天躺平，什么都不做
                    dp[j][i - 1],
                    //  第 i 天卖出，能卖出则说明已买入，max记录第 i-1 天及之前的最大收益，在此基础上收益 prices[i]。
                    // 比如：第二天卖出， 则第一天必然买入（参考 max 的初始值设置）
                    prices[i] + yestodayMaxWhenHold
                )
                // dp[j][i]为最大利润时，必然已经将股票卖出了或第i 天股票价格为0（因为股票价格总是为非负数）
                //
                // 相隔的两天的操作一定不可能为『买入买入』或『卖出卖出』，两次『买入』之间必须间隔至少一次『卖出』
                //
                // 为下次迭代准备（为下次卖出场景准备），更新 yestodayMaxWhenHold 为表示手上持有股票的前提下，i 天的总收益最大值，可能情况：
                // 1）第 i 天不买入，则必须要求i-1天或之前已买买入，即 yestodayMaxWhenHold
                // 2）第 i 天买入， 则在第i-1天的最大利润基础上，买入股票，即减去第 i 天的股票价格（prices[i]）
                // 比较取更大的值
                yestodayMaxWhenHold = maxOf(yestodayMaxWhenHold, dp[j - 1][i - 1] - prices[i])
            }
        }

        return dp[k][prices.size - 1]
    }

    fun maxProfit1(k: Int, prices: IntArray): Int {
        // 从题目示例可以得出：连续的一次买入和一次卖出为一次交易
        if (k < 1) {
            return 0 // 不能买不能卖，最好的策略不买，利润最高0
        }

        // 因为当天买入候随即卖出不会改变当天的利润，所以不考虑这种操作

        // dayCount 天数，等于 prices.size
        val dayCount = prices.size
        if (dayCount <= 1) {
            return 0 // 最好的策略是不买，或者买入当天卖出，利润最高0
        }

        // trasactionCount 交易次数：0，1，2...k+1
        // day 日期：0，1，2...dayCount-1
        // maxProfit[transactionCount][day] 第 day 天最多交易 transactionCount 次的最大利润
        var maxProfit = Array(k + 1) { IntArray(dayCount) }

        // 在买入的情况下第最大的利润
        var maxProfitWhenHold: Int

        // 交易0次，最大利润为0。进行最多1次交易，最大利润为0，已买入的情况下最大利润为 -prices[0]
        // 所以，交易次数从1开始
        for (transactionCount in 1 until k + 1) {
            maxProfitWhenHold = -prices[0]
            // 第0天最大利润为0，所以从第1天开始
            // 每天能执行的操作只有3种：
            // 1）买入
            // 2）卖出
            // 3）不买不卖（或买入即卖出）
            for (day in 1 until dayCount) {
                maxProfit[transactionCount][day] =
                    maxOf(maxProfit[transactionCount][day - 1], prices[day] + maxProfitWhenHold)
                maxProfitWhenHold =
                    maxOf(maxProfitWhenHold, maxProfit[transactionCount - 1][day - 1] - prices[day])
            }
        }

        return maxProfit[k][dayCount - 1]
    }
}

/// 70. 爬楼梯
class Solution70 {
    fun numWays(n: Int): Int {
        // 跳法 ways[n]=ways[n-1]+ways[n-2]
        // 台阶编号 0（代表地面），1，2，3...，n，显然 ways[0]=0, ways[1]=1
        var ways = IntArray(n + 1)
        if (n == 0) return 0
        if (n == 1) return 1
        if (n == 2) return 2
        ways[1] = 1
        ways[2] = 2
        for (i in 3 until n + 1) {
            ways[i] = ways[i - 1] + ways[i - 2]
        }
        return ways[n]
    }
}

class Solution70_20220523 {
    fun climbStairs(n: Int): Int {
        val count = IntArray(n + 1)
        count[0] = 1
        count[1] = 1
        for (i in 2 until n + 1) {
            count[i] = count[i - 1] + count[i - 2]
        }
        return count[n]
    }
}

// 62. 不同路径
class Solution62 {
    fun uniquePaths(m: Int, n: Int): Int {
        // 到达坐标点点总路径数
        // paths[i][j] = paths[i-1][j] + paths[i][j-1]
        var paths = Array(m) { IntArray(n) }
        for (i in 0 until m) {
            for (j in 0 until n) {
                if (i == 0 || j == 0) {
                    paths[i][j] = 1
                } else {
                    paths[i][j] = paths[i - 1][j] + paths[i][j - 1]
                }
            }
        }

        return paths[m - 1][n - 1]
    }
}

/// 64. 最小路径和
class Solution64 {
    fun minPathSum(grid: Array<IntArray>): Int {
        // m 行，n 列，grid[i][j]为非负整数（0<=i<m, 0<=j<n）
        val m = grid.size
        val n = grid.getOrElse(0) { IntArray(0) }.size

        if (m == 0 || n == 0) return 0

        // 路径总和最小值
        // cost[i][j] = maxOf(coast[i-1][j], coast[i][j-1]) + grid[i][j])
        var cost = Array(m) { IntArray(n) }

        // 初始化第一行第一列为 grid[0][0]
        cost[0][0] = grid[0][0]
        // 第一列 从第二行开始，从上往下
        for (i in 1 until m) {
            cost[i][0] = cost[i - 1][0] + grid[i][0]
        }
        // 第一行 从第二列开始，从左往右
        for (j in 1 until n) {
            cost[0][j] = cost[0][j - 1] + grid[0][j]
        }
        // 从第二行第二列开始遍历
        for (i in 1 until m) {
            for (j in 1 until n) {
                cost[i][j] = minOf(cost[i - 1][j], cost[i][j - 1]) + grid[i][j]
            }
        }
        return cost[m - 1][n - 1]
    }
}

/// 72. 编辑距离
class Solution72 {
    fun minDistance(word1: String, word2: String): Int {
        val l1 = word1.length + 1
        val l2 = word2.length + 1

        /// 例子： house => ros
        ///       j
        ///       0 1 2 3
        ///       # r o s
        /// i 0 # 0 1 2 3 => 从空字符串 a 转化到非空字符串 b，编辑距离为 b.length
        ///   1 h 1
        ///   2 o 2
        ///   3 u 3
        ///   4 s 4
        ///   5 e 5
        ///       从非空字符串 b 转化到空字符串 a，编辑距离为 b.length

        /// op[i][j] 代表 word1[0..<i] 转化到 word2[0..<j] 所使用的最少操作次数
        /// word1[0..<i] 代表 word1 中从第 1 个字符开始到第 i 个字符结束（包含第 i 个字符）构成的子字符串, 例如：word1[0..<word1.length] 即为 word1.
        val op = Array(l1) { IntArray(l2) }

        /// 可能操作有3种，插入，删除，替换。一次操作包括操作类型和操作位置。同一组操作按不同顺序执行，得到的结果是一样的。
        /// 所以将问题简化，从 word1 和 word2 的最后一个字符往前推导并进行操作：
        /// 设 i 指向 word1 的最后一个字符，设 j 指向 word2 的最后一个字符。
        ///（1）如果 word1[i] == word2[j]，则 op[i][j] = op[i-1][j-1] (相当于直接将 word1 和 word2 的最后一个字符删除，因为相同所以不需要转化，可以不考虑)
        ///（2）如果 word1[i] ！= word2[j]，则对 word1[0..<i] 可以进行 3 种操作，以保证 word1[0..<i] 转化为 word2[0..<j]，如下：
        ///     1）将字符 word2[j-1] 插入 word1[0..<i] 末尾。相当于先将 word1[0..<i] 转化为 word2[0..<j-1]，所需步数为 op[i][j-1], 然后在末尾插入字符 word2[j-1]，转化为 word2[0..<j]，所需步数为 1。总结一下：
        ///        op[i][j]=op[i][j-1]+1
        ///     2）将字符 word1[i-1] 删除，即 word1[0..<i] 转化为 word1[0..<i-1]，所需步数 1，然后将 word1[0..<i-1] 转化为 word2[0..<j], 所需步数为 op[i-1][j]。总结一下：
        ///        op[i][j]=1+op[i-1][j]
        ///     3）将字符 word1[i-1] 替换成 word2[j-1]，相当于先做字符替换，步数为1，替换后 word1[i-1] 和 word2[j-1] 相等，所以直接忽略，然后将 word1[0..<i-1] 转化为 word2[0..<j-1]，所需步数为 op[i-1][j-1]。总结一下：
        ///        op[i][j]=1+op[i-1][j-1]
        ///     从以上3种情况种选出最小值即为 op[i][j]（最少操作步数）

        for (i in 1 until l1) { // i 代表 word1 的长度，例如：i为1则对应 word1[0..<1], i为2则对应 word2[0..<2]
            op[i][0] = op[i - 1][0] + 1
        }

        for (j in 1 until l2) {
            op[0][j] = op[0][j - 1] + 1
        }

        for (i in 1 until l1) {
            for (j in 1 until l2) {
                if (word1[i - 1] == word2[j - 1]) {
                    op[i][j] = op[i - 1][j - 1]
                } else {
                    op[i][j] = minOf(
                        op[i][j - 1] + 1,
                        op[i - 1][j] + 1,
                        op[i - 1][j - 1] + 1
                    )
                }
            }
        }

        return op[l1 - 1][l2 - 1]
    }

    //// 递归解法 （超时了）

    fun minDistance1(word1: String, word2: String): Int {
        val mem = Array(word1.length + 1) { IntArray(word2.length + 1) { -1 } }
        return minDistance_(word1, word1.length, word2, word2.length, mem)
    }

    fun minDistance_(
        word1: String,
        len1: Int,
        word2: String,
        len2: Int,
        mem: Array<IntArray>
    ): Int {
        if (len1 == 0 || len2 == 0) return maxOf(len1, len2).also { mem[len1][len2] = it }

        val c1 = word1[len1 - 1]
        val c2 = word2[len2 - 1]

        fun getMemOrElseCalculate(l1: Int, l2: Int): Int {
            var r = mem[l1][l2]
            if (r == -1) {
                r = minDistance_(word1, l1, word2, l2, mem)
            }
            return r
        }

        if (c1 == c2) {
            return getMemOrElseCalculate(len1 - 1, len2 - 1)
        }

        // 目标：将 word1 通过编辑转化为 word2，可能操作有插入，删除，替换
        return minOf(
            getMemOrElseCalculate(len1 - 1, len2),
            getMemOrElseCalculate(len1, len2 - 1),
            getMemOrElseCalculate(len1 - 1, len2 - 1)
        ) + 1

    }
}

// 10. 正则表达式匹配
class Solution10 { // 动态规划
    ///  匹配 "*"  和 "."
    fun isMatch(s: String, p: String): Boolean {
        // 给 p 分词，从后往前扫描，
        // 1）如果遇到"*"则将"*"和它的前一个字符打包作为一个匹配项。
        // 2）否则必然遇到的是"a-z"或"."
        // 匹配项数组
        var patterns = ArrayList<String>()
        var i = 0
        while (i < p.length) {
            val hasNext = i + 1 < p.length
            if (hasNext && p[i + 1] == '*') {
                patterns.add("" + p[i] + p[i + 1])
                i++
            } else {
                patterns.add(p[i].toString())
            }
            i++
        }

        // 原字符串前 i 个字符构成的子串是否和匹配项数组前 j 个匹配项组成的匹配字符串匹配
        // +1是因为需要预留一个空字符和空匹配项
        val matches = Array(s.length + 1) { BooleanArray(patterns.size + 1) }

        // 默认空串和空串匹配项匹配
        matches[0][0] = true
        /* // 默认为 false, 可不处理
        for (i in 1 until s.length + 1) {
            matchs[i][0] = false
        }
         */
        // 空字符串和匹配项进行匹配
        for (j in 1 until patterns.size + 1) {
            matches[0][j] = patterns[j - 1].last() == '*' && matches[0][j - 1]
        }

        for (i in 1 until s.length + 1) {
            for (j in 1 until patterns.size + 1) {
                matches[i][j] =
                        //  字符串 s[0..<i-1] 和 patterns[0..<j] 匹配，且当前字符和当前匹配项匹配，并且匹配项时通配符
                    (matches[i - 1][j] && isMatchEmpty(patterns[j - 1]) && isCharMatch(
                        s[i - 1],
                        patterns[j - 1]
                    )) ||
                            // 字符串 s[0..<i-1] 和 patterns[0..<j-1] 匹配，且且当前字符和当前匹配项匹配
                            (matches[i - 1][j - 1] && isCharMatch(s[i - 1], patterns[j - 1])) ||
                            // 字符串 s[0..<i] 和 patterns[0..<j-1] 匹配， 且当前匹配项时通配符（可以匹配空字符串）
                            (matches[i][j - 1] && isMatchEmpty(patterns[j - 1]))
            }
        }

        return matches[s.length][patterns.size]
    }

    // 字符是否和匹配项匹配，匹配项可能是 a-z 或 . 中的一个字符，或者是 a-z 或 . 中的一个字符加一个*, 例如："a*", ".*"
    private fun isCharMatch(c: Char, p: String): Boolean {
        return (p.length == 1 && (c == p[0] || p[0] == '.')) ||
                (p.length == 2 && p[1] == '*' && (c == p[0] || p[0] == '.'))
    }

    private fun isMatchEmpty(p: String): Boolean {
        return p.length == 2 && p[1] == '*'
    }
}

class Solution10_20220521 { // 递归
    // s可能是长度为1的字符串或空字符串
    // pattern长度必须为1或2，例如 "a", "a*", ".*"
    fun stringMatch(s: String, pattern: String): Boolean {
        if (pattern.length == 2) {
            assert(pattern[1] == '*')
            if (s.isEmpty() || pattern[0] == '.' || s[0] == pattern[0]) return true
            return false
        }

        assert(pattern.length == 1)
        return (s.length == 1 && (pattern[0] == '.' || s[0] == pattern[0]))
    }

    fun match(
        s: String,
        p: List<String>,
        mem: MutableMap<String, MutableMap<String, Boolean>>
    ): Boolean {
        fun hasMem(s: String, p: String): Boolean {
            return mem.containsKey(s) && mem[s]!!.containsKey(p)
        }

        fun getMem(s: String, p: String): Boolean {
            if (mem.containsKey(s)) {
                if (mem[s]!!.containsKey(p)) {
                    return mem[s]!![p]!!
                }
            }
            return false
        }

        fun setMem(s: String, p: String, v: Boolean) {
            var map: MutableMap<String, Boolean>
            if (mem.containsKey(s)) {
                map = mem[s]!!
            } else {
                map = mutableMapOf<String, Boolean>()
                mem[s] = map
            }
            map[p] = v
        }

        if (hasMem(s, p.joinToString(""))) {
            return getMem(s, p.joinToString(""))
        }

        if (s.isNotEmpty()) {
            if (p.isEmpty()) {
                return false.also { setMem(s, p.joinToString(""), it) }
            } else {
                // s.isNotEmpty() && p.isNotEmpty()
                val subList = (if (1 < p.size) p.subList(1, p.size) else listOf<String>())
                return (stringMatch("", p[0]) && match(s, subList, mem) ||
                        stringMatch(s.substring(0, 1), p[0]) && (p[0].length == 2 && match(
                    s.substring(1, s.length),
                    p,
                    mem
                ) || match(
                    s.substring(1, s.length),
                    subList,
                    mem
                ))).also { setMem(s, p.joinToString(""), it) }
            }
        } else {
            if (p.isEmpty()) {
                return true.also { setMem(s, p.joinToString(""), it) }
            } else { // s.isEmpty && p.isNotEmpty
                val subList = (if (1 < p.size) p.subList(1, p.size) else listOf<String>())
                return stringMatch(s, p[0]) && match(s, subList, mem).also {
                    setMem(
                        s,
                        p.joinToString(""),
                        it
                    )
                }
            }
        }
    }

    fun isMatch(s: String, p: String): Boolean {
        var patterns = mutableListOf<String>()
        var pre: Char? = null
        for (i in 0 until p.length) {
            if (pre == null) {
                pre = p[i]
            } else if (p[i] == '*') {
                patterns.add(pre.toString() + p[i])
                pre = null
            } else {
                patterns.add(pre.toString())
                pre = p[i]
            }
        }
        if (pre != null) {
            patterns.add(pre.toString())
        }

        val mem = mutableMapOf<String, MutableMap<String, Boolean>>()
        return match(s, patterns, mem)
    }
}

/// 96. 不同的二叉搜索树
/// 给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树（BST）的种数。
class Solution96 {
    fun numTrees(n: Int): Int {
        /*
        思路：
        （1）1 到 n 组成的 n 个数顺序数组，从中任选一个数 i 为根，其左边的数为左子树（有 i-1 个数，假设有 a 种），右边的数为右子树（有 n-i 个数，假设有 b 种）。依次类推对左子树和右子树做同样的操作。
            1）所以，以 i 为根的 BST 一共有 a*b 种。
            2）i 的选择可能有 n 种。
        （2）设 counts[i] (1 <= i <= n) 为 i 个数顺序数组构成的二叉搜索树的种数。 count[n] 即为我们所求的结果。
        （3）设 countOfRoot[i, n] 为以数字 i 为根，节点个数为 n 的 BST 的种数，则 countOfRoot[i, n] = counts[i-1] * counts[n-i]
        （4）i 分别取 1 到 n，计算 countOfRoot[i, n]，然后将所有 countOfRoot[i, n] 加和即为 counts[n]，根据（3）可以得出，
            i 分别取 1 到 n，计算 counts[i-1] * counts[n-i]，所有加和得到结果 counts[n]
        （5）counts[0] = 1, counts[1] = 1
        */
        var counts = IntArray(n + 1)
        counts[0] = 1
        counts[1] = 1
        for (c in 2 until n + 1) {// 计算 c 从 2 到 n，c 为 count 缩写。
            for (r in 1 until c + 1) {// 选 BST 的根，r 为 root 缩写。
                counts[c] += counts[r - 1] * counts[c - r]
            }
        }
        return counts[n]
    }

    /// 卡塔兰数
    fun numTrees1(n: Int): Int {
        var c = 1L
        // c(0) = 1, c(n+1)=c(n)*(2(2n+1)/(n+2))
        for (k in 0 until n) {
            c = c * 2 * (2 * k + 1) / (k + 2)
        }
        return c.toInt();
    }

    fun numTrees2(n: Int): Int {
        // 根据 n 的取值分析结果，观察规律
        // c(0) = 1
        // c(1) = 1
        // c(2) = 2
        // c(3) = 5
        // 符合卡特兰数列规律
        // c(n) = c(n-1)*(4*n-2)/(n+1)
        var c = 1L // c(0)
        for (i in 1 until n + 1) {
            // 求 c(i)
            c = c * (4 * i - 2) / (i + 1)
        }

        // 此时 i 等于 n+1，c 承载 c(n) 的结果

        return c.toInt()
    }
}

/// 946. 验证栈序列
/*
 给定pushed和popped两个序列，每个序列中的 值都不重复，只有当它们可能是在最初空栈上进行的推入 push 和弹出 pop 操作序列的结果时，返回 true；否则，返回 false。
 */
class Solution946 {
    // 思路：模拟进出栈操作
    fun validateStackSequences(pushed: IntArray, popped: IntArray): Boolean {
        var stack = LinkedList<Int>()
        var pushedIndex = 0
        var poppedIndex = 0
        while ((pushedIndex < pushed.size || stack.size > 0) && poppedIndex < popped.size) {
            val top = stack.lastOrNull()
            val pus = pushed.getOrNull(pushedIndex)
            val pop = popped[poppedIndex]
            if (pop == pus) {
                pushedIndex++
                poppedIndex++
            } else if (pop == top) {
                stack.removeLast()
                poppedIndex++
            } else {
                // pop != top && pop != pus
                if (pus != null) {
                    stack.add(pus)
                    pushedIndex++
                } else {
                    return false
                }
            }
        }
        return pushedIndex == pushed.size && stack.size == 0 && poppedIndex == popped.size
    }
}

/// 957. N 天后的牢房
class Solution957 {
    /// 方法1：暴力模拟，超时！！！
    fun prisonAfterNDays(cells: IntArray, n: Int): IntArray {
        // f(n,i) 代表第 n 天第 i 个房间是否有人, n 代表天数（0,1,2,...N-1）,i代表房间号（0，1，2，3，4，5，6，7）
        // f(0,i) = cells[i] （初始值）

        // (1) f(n,0) = 0, if n > 0 第一个房间从第2天开始总是空置，第8个房间同理，f(n,7) = 0, if n > 0
        // (2) f(n,i) = if f(n-1, i-1) == f(n-1, i+1) 1 else 0

        var cells = cells
        // 需要一个临时空间暂存昨天的房间状态
        var yesterdayCells = cells.clone()

        for (n in 1 until n + 1) {
            cells[0] = 0
            cells[cells.size - 1] = 0
            for (i in 1 until cells.size - 1) {
                cells[i] = if (yesterdayCells[i - 1] == yesterdayCells[i + 1]) 1 else 0
            }
            yesterdayCells = cells.also { cells = yesterdayCells }
        }

        return yesterdayCells
    }

    /// 方法2：将数组换成 Int
    /// 假设 cells 共 8 位，则对应的 Int 取值总共有 2^8 = 256 种取值
    fun prisonAfterNDays1(cells: IntArray, n: Int): IntArray {
        // f(n,i) 代表第 n 天第 i 个房间是否有人, n 代表天数（0,1,2,...N-1）,i代表房间号（0，1，2，3，4，5，6，7）
        // f(0,i) = cells[i] （初始值）

        // (1) f(n,0) = 0, if n > 0 第一个房间从第2天开始总是空置，第8个房间同理，f(n,7) = 0, if n > 0
        // (2) f(n,i) = if f(n-1, i-1) == f(n-1, i+1) 1 else 0

        // 返回 1 或 0
        fun getBit(value: Int, bitIndex: Int): Int {
            return (value and (1 shl (bitIndex - 1))) shr (bitIndex - 1)
        }

        // bitValue = 0 or 1
        fun valueWithNewBit(value: Int, bitIndex: Int, bitValue: Int): Int {
            return value or (bitValue shl (bitIndex - 1))
        }

        // 第 0 天的状态
        var rooms = 0
        for (i in cells.indices) {
            rooms = rooms or (cells[i] shl (cells.size - i - 1))
        }

        // rooms 状态 => 天数
        var history = mutableMapOf<Int, Int>()
        var day = 1
        var didJump = false
        while (day <= n) {
            val yesterdayRooms = rooms

            if (!didJump) {
                if (history.containsKey(rooms)) {
                    val historyDay = history[rooms]!!
                    val period = day - historyDay
                    day += (n - day) / period * period // 跨过所有中间循环周期，跳到最后一次循环周期
                    didJump = true
                } else {
                    history[rooms] = day
                }
            }

            // 第一位和最后一位清零
            rooms = 0
            // 给房间重新编号 (和 Int 二进制位对应)
            // 只需遍历第 2 到第 7 位，第 1 和第 8 位已经重置为 0
            // 8 - 7 - 6 - 5 - 4 - 3 - 2 - 1
            for (i in 2 until cells.size) {
                val l = getBit(yesterdayRooms, i - 1)
                val r = getBit(yesterdayRooms, i + 1)
                val bitValue = if (l == r) 1 else 0
                rooms = valueWithNewBit(rooms, i, bitValue)
            }

            day++
        }

        for (i in cells.indices) {
            val shiftCount = cells.size - i - 1
            cells[i] = (rooms and (1 shl shiftCount)) shr shiftCount
        }

        return cells
    }
}

/// 935. 骑士拨号器
class Solution935 {
    fun knightDialer(n: Int): Int {
        val numberCount = 10
        var numberRelations = listOf<IntArray>(
            intArrayOf(4, 6),    // 4 -> 0, 6 -> 0，可以从 4 或 6 跳到 0
            intArrayOf(6, 8),    // 6 -> 1, 8 -> 1
            intArrayOf(7, 9),    // 7 -> 2, 9 -> 2
            intArrayOf(4, 8),    // 4 -> 3, 8 -> 3
            intArrayOf(0, 3, 9), // 0 -> 4, 3 -> 4, 9 -> 4
            intArrayOf(),        // 无 -> 5
            intArrayOf(0, 1, 7), // 0 -> 6, 1 -> 6, 7 -> 6
            intArrayOf(2, 6),    // 2 -> 7, 6 -> 7
            intArrayOf(1, 3),    // 1 -> 8, 3 -> 8
            intArrayOf(2, 4)    // 2 -> 9, 4 -> 9
        )

        // 跳 n 次后落到数字 i 的总共电话号数目
        // counts[i][n] = counts[numberRelations[i][0]][n - 1] + counts[numberRelations[i][1]][n - 1] + ...
        var counts = Array(numberCount) { Array<BigInteger>(n + 1) { BigInteger.valueOf(0) } }

        // 初始化
        for (i in 0 until numberCount) {
            counts[i][1] = BigInteger.valueOf(1)
        }

        // 跳 n 次
        for (jump in 2 until n + 1) {
            for (number in 0 until numberCount) {
                var sum = BigInteger.valueOf(0)
                numberRelations[number].forEach {
                    sum += counts[it][jump - 1]
                }
                counts[number][jump] = sum
            }
        }

        var result = BigInteger.valueOf(0)
        for (i in 0 until numberCount) {
            result += counts[i][n]
        }

        return (result.mod(BigInteger.valueOf(1000000007))).toInt()
    }
}

/// 902. 最大为 N 的数字组合
class Solution902 {
    fun atMostNGivenDigitSet(digitStrings: Array<String>, n: Int): Int {
        var digits = digitStrings.map { it.first() - '0' }.toIntArray()
        var nDigitCount = 0

        // 构造目标数的数字数组
        var nDigits = mutableListOf<Int>()
        var tmpN = n
        while (tmpN > 0) {
            nDigits.add(tmpN % 10)
            tmpN /= 10
            nDigitCount++
        }
        nDigits.reverse()

        // 分2种情况处理

        // 0 < n < 10 的情况, 即个位数
        if (nDigitCount == 1) {
            return digits.count { it <= n }
        }

        /// n >= 10 的情况

        // 记录 1,2,3,...,nDigitCount-1 位数分别可以构造几个
        // count[i] 代表使用 digits 数组中的数可以构造 count[i] 个 i 位数
        var count = IntArray(nDigitCount)
        count[0] = 1
        count[1] = digits.size
        for (i in 2 until nDigitCount) {
            count[i] = count[i - 1] * digits.size
        }

        /// 总个数
        var totalCount = 0
        for (i in 1 until nDigitCount) totalCount += count[i]

        // 此时 totalCount 代表所有可能的 1 到 nDigitCount - 1 位数的总个数

        // 处理可能构造的 nDigitCount 位数
        for (i in nDigits.indices) {
            val nDigit = nDigits[i]
            val index = BinarySearch.indexOfLessThanOrEqual(digits, nDigit)
            if (index == -1) {
                break
            }
            val digit = digits[index]
            if (digit == nDigit) {
                if (i == nDigits.size - 1) {
                    totalCount += 1
                }
                if (index > 0) {
                    totalCount += index * count[nDigits.size - (i + 1)]
                }
            } else {
                // digits.size - (i + 1) 可能为 0
                totalCount += (index + 1) * count[nDigits.size - (i + 1)]
                break
            }
        }

        return totalCount
    }

    /// 官方解法一 动态规划
    fun atMostNGivenDigitSet1(digitStrings: Array<String>, n: Int): Int {
        var digits = digitStrings.map { it.first() - '0' }.toIntArray() // 可选数字的升序排列
        var nDigits = n.toString().map { it - '0' }.toIntArray() // 目标数字的每一位数，从高位到低位顺序的排列

        /**
        设 SubDigit[i] 为 nDigits 后 i 位构成的数字，假设 n = 52033, 则 nDigits = [5, 2, 0, 3, 3]，而 SubDigit[i]
        取值如下：
        - SubDigit[0] 为 空（后续做特殊处理）
        - SubDigit[1] 为 3
        - SubDigit[2] 为 33
        - SubDigit[3] 为 033
        - SubDigit[4] 为 2033
        - SubDigit[5] 为 52033
        显然 SubDigit[nDigits.size] == n，即为目标数

        对 SubDigit[i] 特殊处理一下，变成数组的表达形式，用 SubDigits[i] 表达，则
        SubDigit[i] 和 SubDigits[i] 一一对应。
        - SubDigit[0] 为 空(后续做特殊处理) => SubDigits[0] 为 []
        - SubDigit[1] 为 3               => SubDigits[1] 为 [3]
        - SubDigit[2] 为 33              => SubDigits[2] 为 [3,3]
        - SubDigit[3] 为 033             => SubDigits[3] 为 [0,3,3] （注意这种特殊情况）
        - SubDigit[4] 为 2033            => SubDigits[4] 为 [2,0,3,3]
        - SubDigit[5] 为 52033           => SubDigits[5] 为 [5,2,0,3,3]

        ^_^=_=毫无意义的分隔线=_=^_^

        c[i] 代表是使用 digits 中的数构成位数为 i 位，且小于或等于 SubDigit[i] 的数的总个数。
        根据以上，可以得出所求解的值为：
        c[nDigits.size] + digits.size^1 + digits.size^2 +...+ digits.size^(nDigits.size - 1)

        求解 c 可以从 i 为 0 开始（i代表构造出来的数字的位数）
        - c[0]，i 为 0，则无对应的实际数字，可以理解为从 digits 种选择 0 个数的选法，即只有 1 种，但实际上无法构成一个数，然而为了后续计算方便主动设置 c[0] = 1
        - 例如：已经确定个位数的情况下
        - c[1]，首先找到 SubDigits[1]，取 SubDigits[1].first，设为 target。然后，从 digits 中使用二分查找找到
        最后一个小于等于 target 的数对应的下标（按惯例，未找到时返回-1），设该下标为 targetIndex：
        - 如果 targetIndex == -1，则 c[1] = 0，
        - 说明不可能从 digits 中选出 1 个数来构造出一个小于或等于 SubDigit[1] 的 1 位数。
        - 也可以这么理解，从 [3,4,5] 中选择一个数且要求这个数小于等于 2，则一共有 0 种选法。
        - 如果 targetIndex != -1, 则判断 target 和 digits[targetIndex] 是否相等，根据 targetIndex 的获取方式，
        可以知道，digits[targetIndex] <= target，所以有 2 种情况：
        - digits[targetIndex] == target，则 digits[0] 到 digits[targetIndex] 都可以用来构造数字，有 targetIndex + 1 种方法。
        - digits[targetIndex] < target， 则 digits[0] 到 digits[targetIndex] 都可以用来构造数字，有 targetIndex + 1 种方法。
        总结以上2种情况，c[1] = targetIndex + 1
        - c[2]，首先找到 SubDigits[2]，取 SubDigits[2].first，设为 target。然后，从 digits 中使用二分查找找到
        最后一个小于等于 target 的下标，设为 targetIndex：
        - 如果 targetIndex == -1, 则 c[2] = 0
        - 说明不可能从 digits 中选出 2 个数来构造出一个小于等于 SubDigit[2] 的 2 位数。
        - 如果 targetIndex != -1, 则判断 target 和 digits[targetIndex] 是否相等，根据 targetIndex 的获取方式，
        可以知道，digits[targetIndex] <= target，所以有 2 种情况：
        - digits[targetIndex] == target，
        - 当选择 digits[targetIndex] 来作为十位的数，则用于构造个位数的数不能大于 SubDigits[1] (即 c[1]), 所以有 c[1] 种方法。
        - 当选择 digits 中小于 digits[targetIndex] 的数作为十位数，则一共有 targetIndex 种选法，由于十位数小于目标数的十位数，
        所以个位数只需要从 digits 中任选一个数即可，所以有 targetIndex * digits.size 种方法。
        总计：c[1] + targetIndex * digits.size 种方法。
        - digits[targetIndex] < target，
        - 一共有 targetIndex + 1 种选法构造十位数，由于十位数小于目标数的十位数，所以个位数只需要从 digits 中任选一个数即可，
        所以有 (targetIndex + 1) * digits.size 种方法。
        总计：(targetIndex + 1) * digits.size 种方法。
        - c[3]，首先找到 SubDigits[3]，取 SubDigits[3].first，设为 target。然后，从 digits 中使用二分查找找到
        最后一个小于等于 target 的下标，设为 targetIndex：
        - 如果 targetIndex == -1, 则 c[3] = 0
        - 说明不可能从 digits 中选出 3 个数来构造出一个小于等于 SubDigit[3] 的 3 位数。
        - 如果 targetIndex != -1, 则判断 target 和 digits[targetIndex] 是否相等，根据 targetIndex 的获取方式，
        可以知道，digits[targetIndex] <= target，所以有 2 种情况：
        - digits[targetIndex] == target，
        - 当选择 digits[targetIndex] 来作为百位的数，则问题转化为"从 digits 中选择 2 个数来构造小于或等于 SubDigit[2] 的数"，即 c[2]。
        - 当选择 digits 中小于 digits[targetIndex] 的数作为十位数，则一共有 targetIndex 种选法，由于百位数小于目标数的百位数，
        所以十位和个位数只需要从 digits 中任选一个数即可，所以有 targetIndex * digits.size * digits.size 种方法。
        总计：c[2] + targetIndex * digits.size * digits.size 种方法。
        - digits[targetIndex] < target，
        - 一共有 targetIndex + 1 种选法构造百位数，由于百位数小于目标数的百位数，所以十位和个位数只需要从 digits 中任选一个数即可，
        所以有 (targetIndex + 1) * digits.size * digits.size 种方法。
        总计：(targetIndex + 1) * digits.size * digits.size 种方法。

        - 依次类推...

        - c[i]，首先找到 SubDigits[i]，取 SubDigits[i].first，设为 target。然后，从 digits 中使用二分查找找到
        最后一个小于等于 target 的下标，设为 targetIndex：
        - 如果 targetIndex == -1，则 c[i] = 0，说明不可能从 digits 中选出 i 个数来构造一个小于等于 SubDigit[i] 的 i 位数。
        - 如果 targetIndex != -1，则有2种情况：
        - digits[targetIndex] == target,
        - 当选择 digits[targetIndex] 作为最高位的数，则问题转化为"从 digits 中选择 i - 1 个数来构造小于或等于 SubDigit[i - 1] 的数"，即 c[i-1]
        - 当选择 digits 中小于 digits[targetIndex] 的数作为最高位的数，则一共有 targetIndex 种选法，由于最高位小于目标数的最高位，
        所以后续i-1位数只需要从 digits 中任选一个数即可，所以有 targetIndex * (digits.size^(i-1))  种方法。
        总计：c[i] = c[i-1] + targetIndex * (digits.size^(i-1))
        - digits[targetIndex] < target,
        - 以供有 target + 1 种选法构造最高位的数，由于最高位数小于目标数的最高位数，所以后续 i-1 位数只需要从 digits 总任选一个数即可，
        所以有 (targetIndex + 1)*(digits.size^(i-1)) 种方法。
        总计：c[i] = (targetIndex + 1) * (digits.size^(i-1))

        总结一下，状态转移方程：
        c[i] = 0, if targetIndex == -1
        c[i] = c[i-1] + targetIndex * (digits.size^(i-1)), if targetIndex != -1 && digits[targetIndex] == target  (主动设置 c[0] = 1)
        c[i] = (targetIndex + 1) * (digits.size^(i-1)), if targetIndex != -1 && digits[targetIndex] < target

        讨论：if targetIndex != -1 && digits[targetIndex] == target 的情况，c[1] = c[0] + targetIndex * (digits.size^(i - 1)), 此时 c[0] 实际上代表一种方法，所以主动设置 c[0] = 1

        最后，回到我们求解的值：
        根据以上，可以得出所求解的值为：
        c[nDigits.size] + digits.size^1 + digits.size^2 +...+ digits.size^(nDigits.size - 1)
         */

        // 先计算 digits.size 的 1 次方，2 次方，3 次方，...，(nDigits.size - 1) 次方
        var totalCount = 0
        var powers = IntArray(nDigits.size)
        powers[0] = 1
        for (i in 1 until nDigits.size) {
            powers[i] = powers[i - 1] * digits.size
            totalCount += powers[i] // 计算小于 nDigits.size 位的数的个数
        }

        var c = IntArray(nDigits.size + 1)
        c[0] = 1
        for (i in 1 until nDigits.size + 1) {
            val target = nDigits[nDigits.size - i]
            val targetIndex = BinarySearch.indexOfLessThanOrEqual(digits, target)
            if (targetIndex == -1) {
                c[i] = 0
            } else {
                if (digits[targetIndex] == target) {
                    c[i] = c[i - 1] + targetIndex * powers[i - 1]
                } else {
                    c[i] = (targetIndex + 1) * powers[i - 1]
                }
            }
        }

        totalCount += c[nDigits.size]
        return totalCount
    }

    // 官方解法二 数学方法 映射
    fun atMostNGivenDigitSet2(digitStrings: Array<String>, n: Int): Int {
        var digits = digitStrings.map { it.first() - '0' }.toIntArray()
        var nDigits = n.toString().map { it - '0' }.toIntArray()
        var mappedMaxDigits = IntArray(nDigits.size)

        var useMax = false
        // 先求出最大的那个数
        var i = 0
        while (i < nDigits.size) {
            if (useMax) {
                mappedMaxDigits[i] = digits.size
                i++
                continue
            }

            val target = nDigits[i]
            val targetIndex = BinarySearch.indexOfLessThanOrEqual(digits, target)
            if (targetIndex == -1) {
                useMax = true
                while (i - 1 >= 0) { // 往回遍历
                    i--
                    val target = nDigits[i]
                    val targetIndex = BinarySearch.indexOfLessThanOrEqual(digits, target)
                    if (targetIndex == 0) {
                        mappedMaxDigits[i] = 0 // 将之前添的值清0
                    } else {
                        mappedMaxDigits[i] = targetIndex
                        break
                    }
                    assert(targetIndex != -1)
                }
            } else {
                mappedMaxDigits[i] = targetIndex + 1
                if (digits[targetIndex] < target) {
                    useMax = true
                }
            }

            i++
        }

        // 通过映射，计算出最大的这个数属于第几个数
        var result = 0
        for (d in mappedMaxDigits) {
            result = result * digits.size + d
        }
        return result
    }

    // 动态规划复习
    fun atMostNGivenDigitSet3(digitStrings: Array<String>, n: Int): Int {
        var digits = digitStrings.map { it.first() - '0' }.toIntArray()
        var nDigits = n.toString().map { it - '0' }.toIntArray()

        // 所求结果分为2部分，一部分为数位小于 nDigits.size 的数，一部分为数位等于 nDigits.size 的数.
        // counts[i] 代表通过选择 digits 中的数字来构造 i 位数字，能够构造的个数，即 digits.size ^ i。
        var counts = IntArray(nDigits.size)
        if (nDigits.size > 1) {
            counts[1] = digits.size
            for (i in 2 until nDigits.size) { // i = 2 ~> nDigits.size - 1
                counts[i] = counts[i - 1] * digits.size
            }
        }

        // 数位小于 nDigits.size 的数的个数
        var part1Count = counts.sum()

        // 通过选择 digits 中的数字来构造数字，c[i] 代表这些被构造数字中位数为 i 位，且小于等于『nDigits 后 i 位数构成的数』的个数。
        // 显然，c[1] = {digits 中小于等于 nDigits[nDigits.length - 1] 的数字的个数}
        var c = IntArray(nDigits.size + 1)
        var targetIndex = BinarySearch.indexOfLessThanOrEqual(digits, nDigits[nDigits.size - 1])
        c[1] = if (targetIndex == -1) 0 else targetIndex + 1
        for (i in 2 until nDigits.size + 1) {
            var theCount = 0
            var targetDigit = nDigits[nDigits.size - i]
            targetIndex = BinarySearch.indexOfLessThanOrEqual(digits, targetDigit)
            if (targetIndex == -1) {
                c[i] = 0
            } else if (digits[targetIndex] == targetDigit) {
                c[i] = c[i - 1] + targetIndex * counts[i - 1]
            } else { // digits[targetIndex] < targetDigit
                c[i] = (targetIndex + 1) * counts[i - 1]
            }
        }

        // 根据 c[i] 的定义, 数位等于 nDigits.size 的数的个数即为 c[nDigits.size]
        var part2Count = c[nDigits.size]
        return part1Count + part2Count
    }
}

/// 121. 买卖股票的最佳时机
class Solution121 {
    // 暴力遍历，超时
    fun maxProfit(prices: IntArray): Int {
        var max = 0
        // 第 i 天买入，第 j 天卖出，记录最大利润
        for (i in prices.indices) {
            for (j in i + 1 until prices.size) {
                max = maxOf(max, prices[j] - prices[i])
            }
        }
        return max
    }

    fun maxProfit1(prices: IntArray): Int {
        var minPrice = Int.MAX_VALUE
        var maxProfit = 0
        for (i in prices.indices) {
            if (prices[i] < minPrice) {
                minPrice = prices[i]
            } else if (maxProfit < prices[i] - minPrice) {
                maxProfit = prices[i] - minPrice
            }
        }
        return maxProfit
    }

    // DP
    fun maxProfitDP(prices: IntArray): Int {
        /*
         maxProfitWhenHold[i] 代表第 i 天持有股票的情况下最大的利润（可能为负值）
         maxProfitWhenHold[i] = max(maxProfitWhenHold[i - 1], -prices[i])  => 可以使用一个变量来存储

         profit[i] 代表第 i 天的最大利润
         profit[i] = max(
            profit[i - 1], // 今天不操作，利润和昨天一样
            maxProfitWhenHold[i - 1] + prices[i] // 今天卖出，则今天最大利润为「在今天之前已买入的情况下的最大利润」+ 今天卖出的价格
         )

         profit[0] = 0 第一天只能买入或不买入，则最大利润为 0
         */
        var profit = IntArray(prices.size)
        profit[0] = 0
        var maxProfitWhenHold = -prices[0]
        for (i in 1 until prices.size) {
            profit[i] = maxOf(profit[i - 1], maxProfitWhenHold + prices[i])
            maxProfitWhenHold = maxOf(maxProfitWhenHold, -prices[i])
        }
        return profit[prices.size - 1]
    }

    // DP，优化空间复杂度
    fun maxProfitDPWithOnlyO1Space(prices: IntArray): Int {
        /*
         maxProfitWhenHold[i] 代表第 i 天持有股票的情况下最大的利润（可能为负值）
         maxProfitWhenHold[i] = max(maxProfitWhenHold[i - 1], -prices[i])  => 可以使用一个变量来存储

         profit[i] 代表第 i 天的最大利润
         profit[i] = max(
            profit[i - 1], // 今天不操作，利润和昨天一样
            maxProfitWhenHold[i - 1] + prices[i] // 今天卖出，则今天最大利润为「在今天之前已买入的情况下的最大利润」+ 今天卖出的价格
         )

         profit[0] = 0 第一天只能买入或不买入，则最大利润为 0

         计算 profit[i] 时仅使用到了 profit[i-1] 和 maxProfitWhenHold[i - 1]，profit[i-1]之前 和 maxProfitWhenHold[i - 1]之前的元素不会在被用到，所以可以使用单个变量就可以承载了。
         */
        var profit = 0
        var maxProfitWhenHold = -prices[0]
        for (i in 1 until prices.size) {
            profit = maxOf(profit, maxProfitWhenHold + prices[i])
            maxProfitWhenHold = maxOf(maxProfitWhenHold, -prices[i])
        }
        return profit
    }
}

/// 53. 最大子数组和
class Solution53 {
    /*
     给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。子数组 是数组中的一个连续部分。
     */
    fun maxSubArray(nums: IntArray): Int {
        // s[i] 代表 nums 中以下标 i 元素结尾的连续子数组的最大和
        // s[i] = max(s[i - 1] + nums[i], nums[i])
        // 因为以上计算只会用到 s[i - 1] 可以只用一个变量即可承载
        var s = nums[0]
        var max = nums[0]
        for (i in 1 until nums.size) {
            s = maxOf(s + nums[i], nums[i])
            max = maxOf(max, s)
        }
        return max
    }
}

/// 122. 买卖股票的最佳时机 II
class Solution122 {
    // DP
    fun maxProfit(prices: IntArray): Int {
        // 不限制交易次数

        // ph[i] 代表第 i 天持有股票的情况下最大的利润（可能为负值），ph[0] = -prices[0]
        // ph[i] = max(ph[i-1], p[i-1]-prices[i])
        // 不限交易次数，所以需要添加 p[i-1]
        var ph = IntArray(prices.size)
        ph[0] = -prices[0]

        // p[i] 代表第 i 天能获取的最大利润，
        // p[i] = max(p[i-1], ph[i-1]+prices[i])
        // p[0] = 0
        var p = IntArray(prices.size)

        for (i in 1 until prices.size) {
            p[i] = maxOf(p[i - 1], ph[i - 1] + prices[i])
            ph[i] = maxOf(ph[i - 1], p[i] - prices[i])
        }

        return p[prices.size - 1]
    }

    // 贪心
    fun maxProfit1(prices: IntArray): Int {
        var max = 0
        for (i in 1 until prices.size) {
            max += maxOf(0, prices[i] - prices[i - 1])
        }
        return max
    }
}

/// 剑指 Offer II 062. 实现前缀树
class SolutionJZOfferII062 {
    class Trie() {
        /** Initialize your data structure here. */
        class TrieNode() {
            var count: Int = 0
            val next = MutableList<TrieNode?>(26) { null }
        }

        val root = MutableList<TrieNode?>(26) { null }

        /** Inserts a word into the trie. */
        fun insert(word: String) {
            var entries = root
            for (i in word.indices) {
                val c = word[i]
                val index = c - 'a'
                var node = entries[index]
                if (node == null) {
                    node = TrieNode()
                    entries[index] = node
                }
                if (i == word.length - 1) {
                    node.count += 1
                }
                entries = node.next
            }
        }

        /** Returns if the word is in the trie. */
        fun search(word: String): Boolean {
            var entries = root
            for (i in word.indices) {
                val c = word[i]
                val index = c - 'a'
                val node = entries[index]
                if (node != null) {
                    if (i == word.length - 1 && node.count > 0) {
                        return true
                    }
                    entries = node.next
                } else {
                    return false
                }
            }
            return false
        }

        /** Returns if there is any word in the trie that starts with the given prefix. */
        fun startsWith(prefix: String): Boolean {
            var entries = root
            for (i in prefix.indices) {
                val c = prefix[i]
                val index = c - 'a'
                val node = entries[index]
                if (node != null) {
                    entries = node.next
                } else {
                    return false
                }
            }
            return true
        }

    }

    /**
     * Your Trie object will be instantiated and called as such:
     * var obj = Trie()
     * obj.insert(word)
     * var param_2 = obj.search(word)
     * var param_3 = obj.startsWith(prefix)
     */
}

/// 887. 鸡蛋掉落
class Solution887 {
    fun superEggDrop(K: Int, N: Int): Int {
        // d[k][n] 代表有 k 个鸡蛋，n 层楼时，测试出 f 的最小次数
        // 设置 x 为实际的 f 值，则 x 可能为 1 <= x <= n
        // dp[k][n] = 1 + min(max(dp[k][n - 1], dp[k-1][n-1]), max(dp[k][], dp[k-1][]), max(dp[k][], dp[k-1][]), ...)
        var d = Array(K + 1) { IntArray(N + 1) }

        for (n in 1 until N + 1) {
            d[1][n] = n
        }

        for (k in 1 until K + 1) {
            d[k][0] = 0 // 如果仅有0层楼，则不用测试，所以 f == 0
            d[k][1] = 1
        }

        for (k in 2 until K + 1) {
            for (n in 2 until N + 1) {
                var l = 1
                var r = n
                while (l + 1 < r) {
                    val mid = (l + r) ushr 1
                    val left = d[k - 1][mid - 1] // left 相对 mid 单调递增
                    val right = d[k][n - mid]    // right 相对 mid 单调递减
                    // 所以随着 mid 从 1 到 n 总存在一个点（或2个点）left 和 right 的差值最小
                    if (left < right) {
                        l = mid
                    } else if (left > right) {
                        r = mid
                    } else {
                        l = mid
                        r = mid
                    }
                }
                d[k][n] = 1 + minOf(
                    maxOf(d[k - 1][l - 1], d[k][n - l]),
                    maxOf(d[k - 1][r - 1], d[k][n - r])
                )
            }
        }

        return d[K][N]
    }

    // 默写一遍
    fun superEggDrop1(K: Int, N: Int): Int {
        // d[k][n] 有 k 个鸡蛋，n 层楼时，确认 f 最小的操作次数
        // d[k][n] = 1 + min(
        //    max(d[k][n - 1], d[k - 1][1 - 1]),
        //    max(d[k][n - 2], d[k - 1][2 - 1]),
        //    max(d[k][n - 3], d[k - 1][3 - 1]),
        //    ...
        //    max(d[k][n - x], d[k - 1][x - 1]),
        //    ...
        //    max(d[k][n - n], d[k - 1][n - 1]),
        // )
        //
        // 当 k 和 n 确定时，
        //    (1) d[k][n - x] 随着 x 增大而减小，
        //    (2) 而 d[k - 1][x - 1] 随着 x 增大而增大。 (1 <= x <= n)
        // 所以必然存在1个或2个x使得 d[k][n - x] 和 d[k - 1][x - 1] 差值最小，只需找到这个 x，然后计算并取得最小值即可
        // 基于 d[k][n - x] 和 d[k - 1][x - 1] 遂 x 增大而相互逼近的特性，可以通过二分法来快速查找 x 值

        var d = Array(K + 1) { IntArray(N + 1) }

        // 1 层楼，k 个鸡蛋，总是只需要 1 次测试
        for (k in 1 until K + 1) {
            d[k][1] = 1
            d[k][0] = 0 // 0 层楼时无需测试
        }

        // n 层楼，1 个鸡蛋，需要 n 次测试
        for (n in 1 until N + 1) {
            d[1][n] = n
        }

        for (k in 2 until K + 1) {
            for (n in 2 until N + 1) {
                // 二分查找 x 值
                var l = 1
                var r = n
                while (l + 1 < r) {
                    val mid = (l + r) ushr 1
                    val increase = d[k - 1][mid - 1]
                    val decrease = d[k][n - mid]
                    if (decrease > increase) {
                        l = mid
                    } else if (decrease < increase) {
                        r = mid
                    } else {
                        l = mid
                        r = mid // 直接找到了相交的那个点对应的 x 值
                    }
                }

                d[k][n] = 1 + minOf(
                    maxOf(d[k][n - l], d[k - 1][l - 1]),
                    maxOf(d[k][n - r], d[k - 1][r - 1])
                )
            }
        }

        return d[K][N]
    }
}

/// 15. 三数之和
class Solution15 {
    fun threeSumXX(nums: IntArray): List<List<Int>> {
        if (nums.size < 3) return listOf()

        nums.sort()

        val result = arrayListOf<List<Int>>()

        if (nums[0] + nums[1] + nums[2] > 0) return result
        if (nums[nums.size - 1] + nums[nums.size - 2] + nums[nums.size - 3] < 0) return result

        for (a in 0 until nums.size - 2) {
            if (a > 0 && nums[a] == nums[a - 1]) continue

            if (nums[a] + nums[a + 1] + nums[a + 2] > 0) break
            if (nums[a] + nums[nums.size - 1] + nums[nums.size - 2] < 0) continue

            var c = nums.size - 1
            val target = -nums[a]
            for (b in a + 1 until nums.size - 1) {
                if (b > a + 1 && nums[b] == nums[b - 1]) continue

                if (nums[b] + nums[b + 1] > target) break
                if (nums[b] + nums[c] < target) continue

                while (b < c && nums[b] + nums[c] > target) {
                    c--
                }
                if (b == c) break;
                if (nums[b] + nums[c] == target) {
                    result.add(arrayListOf<Int>(nums[a], nums[b], nums[c]))
                }
            }
        }

        return result
    }

    fun threeSumX(nums: IntArray): List<List<Int>> {
        nums.sort()

        val result = arrayListOf<List<Int>>()

        for (a in 0 until nums.size - 2) {
            if (a > 0 && nums[a] == nums[a - 1]) continue

            if (nums[a] + nums[a + 1] + nums[a + 2] > 0) break
            if (nums[nums.size - 1] + nums[nums.size - 2] + nums[nums.size - 3] < 0) break

            var c = nums.size - 1
            val target = -nums[a]
            for (b in a + 1 until nums.size - 1) {
                if (b > a + 1 && nums[b] == nums[b - 1]) continue

                if (nums[b] + nums[b + 1] > target) break
                if (nums[b] + nums[c] < target) continue

                while (b < c && nums[b] + nums[c] > target) {
                    c--
                }
                if (b == c) break;
                if (nums[b] + nums[c] == target) {
                    result.add(arrayListOf<Int>(nums[a], nums[b], nums[c]))
                }
            }
        }

        return result
    }

    fun threeSum(nums: IntArray): List<List<Int>> {
        nums.sort()

        val result = arrayListOf<List<Int>>()

        for (a in 0 until nums.size - 2) {
            if (a > 0 && nums[a] == nums[a - 1]) continue
            var c = nums.size - 1
            val target = -nums[a]
            for (b in a + 1 until nums.size - 1) {
                if (b > a + 1 && nums[b] == nums[b - 1]) continue
                while (b < c && nums[b] + nums[c] > target) {
                    c--
                }
                if (b == c) break;
                if (nums[b] + nums[c] == target) {
                    result.add(arrayListOf<Int>(nums[a], nums[b], nums[c]))
                }
            }
        }

        return result
    }

    fun threeSum3(nums: IntArray): List<List<Int>> {
        fun bs(nums: IntArray, target: Int, start: Int, end: Int): Int {
            var l = start
            var r = end

            while (l < r) {
                val m = (l + r) ushr 1
                if (nums[m] < target) {
                    l = m + 1
                } else {
                    r = m
                }
            }

            if (l < nums.size && nums[l] == target) return l

            return -1
        }

        nums.sort()

        val result = mutableListOf<List<Int>>()

        for (a in 0 until nums.size - 2) {
            if (a > 0 && nums[a] == nums[a - 1]) continue
            var cc = nums.size - 1
            for (b in a + 1 until nums.size - 1) {
                if (b > a + 1 && nums[b] == nums[b - 1]) continue
                val c = bs(nums, -(nums[a] + nums[b]), b + 1, cc)
                if (c != -1) {
                    cc = c - 1
                    result.add(listOf<Int>(nums[a], nums[b], nums[c]))
                }
            }
        }

        return result
    }

    fun threeSum2(nums: IntArray): List<List<Int>> {
        nums.sort()

        var result = mutableListOf<List<Int>>()

        // a, b, c 为三个下标
        val n = nums.size
        var a = 0
        while (a < n - 2) {
            if (a > 0 && nums[a] == nums[a - 1]) {
                a++
                continue
            }

            val target = -nums[a]
            var b = a + 1
            var c = n - 1
            while (b < c) {
                val sum = nums[b] + nums[c]
                if (sum == target) {
                    result.add(listOf(nums[a], nums[b], nums[c]))
                    while (b < c && nums[b] == nums[b + 1]) b++
                    b++
                    while (b < c && nums[c] == nums[c - 1]) c--
                    c--
                } else if (sum < target) {
                    b++
                } else { // sum > target
                    c--
                }
            }
            a++
        }

        return result
    }

    fun threeSum1(nums: IntArray): List<List<Int>> {
        var result = mutableListOf<List<Int>>()
        nums.sort()
        var first: Int? = null
        for (i in nums.indices) {
            if (first == nums[i]) {
                continue
            }
            first = nums[i]
            result.addAll(threeSumWithFirst(nums, i + 1, first))
        }
        return result
    }

    fun threeSumWithFirst(nums: IntArray, start: Int, first: Int): List<List<Int>> {
        val sum = -first
        var result = mutableListOf<List<Int>>()

        var l = start
        var r = nums.size - 1
        while (l < r) {
            val second = nums[l]
            val third = nums[r]
            if (second + third == sum) {
                result.add(listOf(first, second, third))
                l++
                while (l < r && nums[l] == second) {
                    l++
                }
            } else if (second + third < sum) {
                l++
                while (l < r && nums[l] == second) {
                    l++
                }
            } else { // second + third > sum
                r--
                while (l < r && nums[r] == third) {
                    r--
                }
            }
        }
        return result
    }
}

/// 877. 石子游戏
class Solution877 {
    // 暴力递归
    fun stoneGame(piles: IntArray): Boolean {
        return stoneGameAliceGain(piles, 0, piles.size - 1) > 0
    }

    fun stoneGameAliceGain(piles: IntArray, start: Int, end: Int): Int {
        if (end - start == 1) {
            return maxOf(piles[start] - piles[end], -piles[start] + piles[end])
        }

        return maxOf(
            maxOf(
                piles[start] - piles[end] + stoneGameAliceGain(
                    piles,
                    start + 1,
                    end - 1
                ), // A 选 start，B 选 end
                piles[start] - piles[start + 1] + stoneGameAliceGain(
                    piles,
                    start + 2,
                    end
                ) // A 选 start, B 选 start + 1
            ),
            maxOf(
                piles[end] - piles[start] + stoneGameAliceGain(
                    piles,
                    start + 1,
                    end - 1
                ), // A 选 end, B 选 start
                piles[end] - piles[end - 1] + stoneGameAliceGain(
                    piles,
                    start,
                    end - 2
                )// A 选 end, B 选 end - 1
            )
        )
    }

    // DP
    fun stoneGameDP(piles: IntArray): Boolean {
        // score[i][j] 代表在以 piles 第 i 个元素到第 j 个元素组成的子数组中 Alice 可以获得的石子总数减去 Bob 可
        // 以获得的石子总数的差值
        /*
         score[i][j] = max(
           piles[i] - piles[j] + score[i + 1][j - 1] // Alice 选择了 i，Bob 选择了 j
           piles[i] - piles[i] + score[i + 2][j]     // Alice 选择了 i，Bob 选择了 i+1
           piles[j] - piles[i] + score[i + 1][j - 1] // Alice 选择了 j，Bob 选择了 i
           piles[j] - piles[j + 1] + score[i + 1][j - 1] // Alice 选择了 j，Bob 选择了 j-1
         )

         0 <= i <= piles.size - 2
         1 <= j <= piles.size - 1

         */
        var score = Array(piles.size - 1) { IntArray(piles.size) }

        for (i in 0 until piles.size - 1) {
            // piles 子数组长度为 2
            val tmp = piles[i] - piles[i + 1]
            score[i][i + 1] = if (tmp < 0) -tmp else tmp
        }

        for (l in 3 until piles.size step 2) { // piles 子数组长度为 4，6，8，...
            for (j in piles.size - 1 downTo l) {
                val i = j - l
                var max = Int.MIN_VALUE
                listOf(
                    piles[i] - piles[j] + score[i + 1][j - 1], // Alice 选择了 i，Bob 选择了 j
                    piles[i] - piles[i + 1] + score[i + 2][j], // Alice 选择了 i，Bob 选择了 i+1
                    piles[j] - piles[i] + score[i + 1][j - 1], // Alice 选择了 j，Bob 选择了 i
                    piles[j] - piles[j - 1] + score[i][j - 2]  // Alice 选择了 j，Bob 选择了 j-1
                ).forEach { if (it > max) max = it }
                score[i][j] = max
            }
        }

        return score[0][piles.size - 1] > 0
    }
}

/// 486. 预测赢家
class Solution486 {
    // 递归
    fun PredictTheWinner(nums: IntArray): Boolean {
        return totalScore(nums, 0, nums.size - 1, 1) >= 0
    }

    fun totalScore(nums: IntArray, start: Int, end: Int, turn: Int): Int {
        if (start == end) {
            return nums[start] * turn
        }
        val useStart = nums[start] * turn + totalScore(nums, start + 1, end, -turn)
        val useEnd = nums[end] * turn + totalScore(nums, start, end - 1, -turn)
        return maxOf(useStart * turn, useEnd * turn) * turn
    }

    // DP
    fun PredictTheWinnerDP(nums: IntArray): Boolean {
        // dp[i][j]表示剩余 nums[i..j] 时，先手选手能够获得的最大净胜分
        // dp[i][j] = maxOf(nums[i] - dp[i+1][j], nums[j] - dp[i][j-1])
        // dp[i][i] = nums[i]
        // i <= j
        // 求 dp[0][nums.size-1] >= 0
        var dp = Array(nums.size) { IntArray(nums.size) }
        for (i in 0 until nums.size) {
            dp[i][i] = nums[i]
        }
        for (l in 1 until nums.size) {
            for (i in 0 until nums.size) {
                var j = i + l
                if (j >= nums.size) break
                dp[i][j] = maxOf(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1])
            }
        }

        return dp[0][nums.size - 1] >= 0
    }

    // DP 空间优化
    fun PredictTheWinnerDPMemO_N(nums: IntArray): Boolean {
        // dp[i][j] 代表剩余 nums[i...j] 时，先手选手能获得的最大净胜分
        // dp[i][j] = max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1])
        // 优化内存空间，dp[i][j] 的计算仅用到 dp[i + 1][j] 和 dp[i][j - 1] 使用一维数组代替二维数组
        //
        // dp[i][i] = i
        //          设 n = nums.size - 1
        //          横向 i in 0 ~> n
        //  纵向i    ... dp[i][j - 1]  dp[i][j]     ...
        //  从0到    ...               dp[i + 1][j] ...
        //  n
        //

        var dp = IntArray(nums.size) { nums[it] }

        for (l in 2 until dp.size + 1) {
            for (i in 0 until dp.size - (l - 1)) {
                val j = i + (l - 1)
                // dp[i][j] = maxOf(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1])
                dp[i] = maxOf(nums[i] - dp[i + 1], nums[j] - dp[i])
            }
        }

        return dp[0] >= 0
    }

    /// 递归
    fun PredictTheWinner__(nums: IntArray): Boolean {
        var mem = Array(nums.size) { IntArray(nums.size) { Int.MIN_VALUE } }
        return totalScoreDifference(mem, nums, 0, nums.size - 1) >= 0
    }

    // 代表当前先手的选手可以获得的最大『净胜分』
    fun totalScoreDifference(mem: Array<IntArray>, nums: IntArray, start: Int, end: Int): Int {
        if (mem[start][end] != Int.MIN_VALUE) {
            return mem[start][end]
        }

        var result: Int
        if (start == end) {
            result = nums[start]
        } else {
            result = maxOf(
                nums[start] - totalScoreDifference(mem, nums, start + 1, end),
                nums[end] - totalScoreDifference(mem, nums, start, end - 1)
            )
        }

        mem[start][end] = result
        return result
    }
}

/// 1406. 石子游戏 III
class Solution1406 {
    /// 记忆化递归
    fun stoneGameIII(stoneValue: IntArray): String {
        var mem = IntArray(stoneValue.size) { Int.MIN_VALUE }
        val netScore = total(stoneValue, 0, mem)
        if (netScore > 0) return "Alice"
        if (netScore < 0) return "Bob"
        return "Tie"
    }

    fun total(stoneValue: IntArray, start: Int, mem: IntArray): Int {
        if (mem[start] != Int.MIN_VALUE) {
            return mem[start]
        }

        val end = stoneValue.size - 1

        ///////// 仅剩1堆，只能取走1堆
        if (start == end) {
            return stoneValue[start].also { mem[start] = it }
        }

        // 取走1堆
        val get1 = stoneValue[start] - total(stoneValue, start + 1, mem)

        ///////// 仅剩2堆，只能取走1或2堆
        if (start + 1 == end) {
            // 取走2堆
            val get2 = stoneValue[start] + stoneValue[start + 1] - 0
            return maxOf(get1, get2).also { mem[start] = it }
        }

        // 取走2堆
        val get2 = stoneValue[start] + stoneValue[start + 1] - total(stoneValue, start + 2, mem)

        ///////// 仅剩3堆，只能取走1或2或3堆
        if (start + 2 == end) {
            // 取走3堆
            val get3 = stoneValue[start] + stoneValue[start + 1] + stoneValue[start + 2] - 0
            return maxOf(maxOf(get1, get2), get3).also { mem[start] = it }
        }

        ///////// 剩下超过3堆，只能取走1或2或3堆
        val get3 = stoneValue[start] + stoneValue[start + 1] + stoneValue[start + 2] - total(
            stoneValue,
            start + 3,
            mem
        )
        return maxOf(maxOf(get1, get2), get3).also { mem[start] = it }
    }

    /// 代码整理优化
    fun total1(stoneValue: IntArray, start: Int, mem: IntArray): Int {
        fun getStoneValue(index: Int): Int {
            return stoneValue.getOrElse(index) { 0 }
        }

        // 越界时返回 0
        if (start >= stoneValue.size) return 0

        // 读缓存
        if (mem[start] != Int.MIN_VALUE) {
            return mem[start]
        }

        // 取走1堆
        val get1 = getStoneValue(start) - total(stoneValue, start + 1, mem)
        // 取走2堆
        val get2 =
            getStoneValue(start) + getStoneValue(start + 1) - total(stoneValue, start + 2, mem)
        // 取走3堆
        val get3 =
            getStoneValue(start) + getStoneValue(start + 1) + getStoneValue(start + 2) - total(
                stoneValue,
                start + 3, mem
            )

        // 取最大值，并缓存
        return maxOf(maxOf(get1, get2), get3).also { mem[start] = it }
    }

    /// DP
    fun stoneGameIIIDP(stoneValue: IntArray): String {
        fun getStoneValue(index: Int): Int {
            return stoneValue.getOrElse(index) { 0 }
        }
        // dp[i] = maxOf (
        //   stoneValue[i] - dp[i+1],
        //   stoneValue[i] + stoneValue[i+1] - dp[i+2]
        //   stoneValue[i] + stoneValue[i+1] + stoneValue[i+2] - dp[i+3]
        // )
        val n = stoneValue.size
        var dp = IntArray(n + 3)

        for (i in n - 1 downTo 0) {
            dp[i] = maxOf(
                maxOf(
                    getStoneValue(i) - dp[i + 1],
                    getStoneValue(i) + getStoneValue(i + 1) - dp[i + 2]
                ),
                getStoneValue(i) + getStoneValue(i + 1) + getStoneValue(i + 2) - dp[i + 3]
            )
        }

        if (dp[0] > 0) return "Alice"
        if (dp[0] < 0) return "Bob"
        return "Tie"
    }
}

/// 293. 翻转游戏
class Solution293 {
    fun generatePossibleNextMoves(currentState: String): List<String> {
        var preIsPlush = false
        var result = mutableListOf<String>()
        for (i in currentState.indices) {
            val currentIsPlus = currentState[i] == '+'
            if (preIsPlush && currentIsPlus) {
                result.add(currentState.replaceRange(i - 1 until i + 1, "--"))
            }
            preIsPlush = currentIsPlus
        }
        return result
    }
}

/// 292. Nim 游戏
class Solution292 {
    fun canWinNim(n: Int): Boolean {
        return n % 4 != 0
    }
}

/// 294. 翻转游戏 II
class Solution294 {
    /// 递归
    fun canWin(currentState: String): Boolean {
        var mem = mutableMapOf<String, Boolean>()
        return canWin_(currentState, mem)
    }

    fun canWin_(currentState: String, mem: MutableMap<String, Boolean>): Boolean {
        if (mem.containsKey(currentState)) {
            return mem[currentState]!!
        }

        val n = currentState.length
        for (i in currentState.indices) {
            // 找到一个 ++
            val iCanWin = currentState[i] == '+' && i + 1 < n && currentState[i + 1] == '+'
            if (iCanWin) {
                // 翻转 ++
                val next = currentState.replaceRange(i until i + 2, "--")
                val nextCannotWin = !canWin_(next, mem).also { mem[next] = it }
                if (nextCannotWin) {
                    return true
                }
            }
        }
        return false.also { mem[currentState] = it }
    }

    /// 状态 A 的 SG 数：SG(A) = mex({SG(B) | A -> B})，状态A的SG数是所有状态A的次态的SG数集合的 mex 值。
    /// mex 为 minimum excluded value 的简写, 即 mex(S) 表示集合 S 所不包含的最小非负整数（该值可能比 S 中最大值更大，也可能比 S 中最小值小）。
    ///   例如： mex({2,3}) = 0，mex({3}) = 0，mex({1,2}) = 3
    /// B 是 A 的次态，即从状态 A 可以转移到状态 B。
    fun canWinSG(currentState: String): Boolean {
        var subStates = mutableListOf<Int>() // 记录所有连续 "+" 号子串的长度

        // 从当前状态字符串中提取所有 子状态（注意：子状态和次态是不一样的两个概念）
        var plusLength = 0
        var maxPlusLength = 0
        for (c in "$currentState-") {
            if (c == '+') {
                plusLength++
            } else {
                if (plusLength > 0) {
                    subStates.add(plusLength)
                    if (plusLength > maxPlusLength) {
                        maxPlusLength = plusLength
                    }
                }
                plusLength = 0
            }
        }

        if (maxPlusLength <= 1) return false // 所有子状态都是无法翻转的，所以判定先手输

        // 需要求出 sg(1) ~ sg(maxPlusLength)
        // sg[i] 代表仅由 '+' 号组成的长度为 i 的字符串的 SG 数
        // 根据翻转游戏规则，默认 sg[0] = 0（空字符串无法翻转，所以不存在次态）, sg[1] = 0
        var sg = IntArray(maxPlusLength + 1)

        for (length in 2 until maxPlusLength + 1) { // 长度从 2 到 maxPlusLength，从小到大
            // 仅由 '+' 号组成的长度为 i 的字符串 => 求它的次态的 sg 值的集合 nextStateSgSet，然后找到 mex(nextStateSgSet)
            var nextStateSgSet = mutableSetOf<Int>()
            var nextStateSgMax = 0
            for (j in 0 until length / 2) { // 只需包含前半部分即可，后半部分和前半部分实际为等价的
                /* 每种翻转后，形成的次态可以分解成两种状态 */
                /* 可分解的状态（g值）等于各分解子状态（g值）的异或和 */
                val nextStateSg = sg[j] xor sg[length - j - 2]
                nextStateSgSet.add(nextStateSg)
                nextStateSgMax = maxOf(nextStateSgMax, nextStateSg)
            }
            // 寻找 mex(nextStateSgSet)，不被 nextStateSgSet 包含的最小非负整数
            for (nonnegative in 0 until nextStateSgMax + 2) {
                if (nextStateSgSet.contains(nonnegative)) {
                    continue
                }
                sg[length] = nonnegative // 记录 sg 值
                break
            }
        }

        var result = 0
        subStates.forEach { result = result xor sg[it] }
        return result > 0 // SG数为正则先手必胜，SG为0则后手必胜
    }
}

/// 464. 我能赢吗
class Solution464 {

    /////////////// 方法一：递归 + 状态记忆 ///////////////

    /// 记忆化递归
    fun canIWin(n: Int, t: Int): Boolean {
        // 总和比目标数字小，则先手和后手都无法达到目标数字，判定先手输
        if (n * (n + 1) / 2 < t) return false

        // 如果选完所有数字有可能达到目标数字，即在当前数字数组 numbers 和目标数 target 的条件下，在有限步数内
        // 必然能够决出胜负
        var mem = mutableMapOf<List<Int>, Boolean>()
        return canIWin_((0 until n).map { it + 1 }, t, mem)
    }

    // 如果选完所有数字有可能达到目标数字，即在当前数字数组 numbers 和目标数 target 的条件下，在有限步数内
    // 必然能够决出胜负
    fun canIWin_(numbers: List<Int>, target: Int, mem: MutableMap<List<Int>, Boolean>): Boolean {
        if (mem.containsKey(numbers)) {
            return mem[numbers]!!
        }

        for (i in numbers.indices) {
            if (numbers[i] >= target) {
                return true.also {
                    mem[numbers] = it
                }
            }

            val subTarget = target - numbers[i]
            var subNumbers =
                numbers.filterIndexed { index, _ -> index != i }
            if (!canIWin_(subNumbers, subTarget, mem)) {
                return true.also {
                    mem[numbers] = it
                }
            }
        }

        return false.also {
            mem[numbers] = it
        }
    }

    /////////////// 动态规划，参考 https://leetcode.cn/problems/can-i-win/solution/shuang-zhong-forxun-huan-dpkan-bu-dong-n-wwyj/

    /// DP
    fun canIWin1(n: Int, target: Int): Boolean {
        // 总和比目标数字小，则先手和后手都无法达到目标数字，判定先手输
        if (n * (n + 1) < target) return false

        // 总和小于等于目标数字，则先手和后手必然有一方可以赢
        // maxState 低位 n 个 bit 对应 1 ~ n 这 n 个数字:
        //   - 右起第 1 位（1左移0位）对应数字 1，
        //   - 右起第 k+1 位（1左移k位）对应数字 k+1。
        // 每个 bit 位可能位 1 或 0，为 1 代表选择了该位对应的数字，位 0 代表未选择该位代表的数字
        // 所以所求结果即为：dp[0] （即全部未选时的状态）
        var maxState = 1 shl n
        var dp = BooleanArray(maxState) { false }

        for (state in maxState - 1 downTo 0) {
            var total = target
            for (k in 0 until n) {
                if (((1 shl k) and state) > 0) {
                    // 右起第 k + 1 位对应数字 k + 1, bit 位为 1，代表已选该数字，则直接从目标值中减去该值
                    total -= k + 1
                }
            }

            for (k in 0 until n) {
                if (((1 shl k) and state) > 0) continue // 跳过已被选择过的数字

                // 选择右起第 k + 1 位，即数字 k + 1
                if (k + 1 >= total || !dp[state or (1 shl k)]) {
                    dp[state] = true
                }
            }
        }

        return dp[0]
    }

    ////////////// 结合位运算 + 递归 + 状态记忆，降低内存占用

    /// 记忆化递归 + 位运算
    fun canIWin2(n: Int, t: Int): Boolean {
        // 总和比目标数字小，则先手和后手都无法达到目标数字，判定先手输
        if (n * (n + 1) / 2 < t) return false

        // 如果选完所有数字有可能达到目标数字，即在当前数字数组 numbers 和目标数 target 的条件下，在有限步数内
        // 必然能够决出胜负
        var mem = mutableMapOf<Int, Boolean>()
        // numberState 从第位开始，第 i 位代表数字 i, 第 i 位取值 0 代表该位对应的数字未被选择，反之 1 代表已被选择
        // 所以初始输入为 0
        return canIWin_2(0, n, t, mem)
    }

    // 如果选完所有数字有可能达到目标数字，即在当前数字数组 numbers 和目标数 target 的条件下，在有限步数内
    // 必然能够决出胜负
    // numberState 从第位开始，第 i 位代表数字 i, 第 i 位取值 0 代表该位对应的数字未被选择，反之 1 代表已被选择
    fun canIWin_2(numbersState: Int, n: Int, target: Int, mem: MutableMap<Int, Boolean>): Boolean {
        if (mem.containsKey(numbersState)) {
            return mem[numbersState]!!
        }

        for (i in 0 until n) { // 遍历 n 位
            if ((1 shl i) and numbersState > 0) {
                // 说明在 numberState 状态下，第 i + 1 位（即数字 i + 1）已被选择过，不能继续选择，所以跳过
                continue
            }

            val currentNumber = i + 1
            if (currentNumber >= target) {
                return true.also {
                    mem[numbersState] = it
                }
            }

            val subTarget = target - currentNumber
            // 下一个状态即为选掉当前数字后的状态
            val subNumbersState = numbersState or (1 shl i)
            if (!canIWin_2(subNumbersState, n, subTarget, mem)) {
                return true.also {
                    mem[numbersState] = it
                }
            }
        }

        return false.also {
            mem[numbersState] = it
        }
    }
}

class Solution464_20220522 {
    val mem = mutableMapOf<Int, MutableMap<Int, Boolean>>()
    fun hasMem(desiredTotal: Int, remainNumberBits: Int): Boolean {
        return mem.containsKey(desiredTotal) && mem[desiredTotal]!!.containsKey(remainNumberBits)
    }

    fun getMem(desiredTotal: Int, remainNumberBits: Int): Boolean {
        assert(hasMem(desiredTotal, remainNumberBits))
        return mem[desiredTotal]!![remainNumberBits]!!
    }

    fun setMem(desiredTotal: Int, remainNumberBits: Int, canWin: Boolean) {
        var map: MutableMap<Int, Boolean>
        if (mem.containsKey(desiredTotal)) {
            map = mem[desiredTotal]!!
        } else {
            map = mutableMapOf<Int, Boolean>()
            mem[desiredTotal] = map
        }
        map[remainNumberBits] = canWin
    }

    fun canWin(desiredTotal: Int, remainNumberBits: Int, maxChoosableInteger: Int): Boolean {
        if (hasMem(desiredTotal, remainNumberBits)) {
            return getMem(desiredTotal, remainNumberBits)
        }

        // remainNumberBits 不可能会等于 0，因为在调用前做了限制

        for (n in maxChoosableInteger downTo 1) {
            // n 即为数字 (从大到小遍历)
            // 通过判断 remainNumberBits 中第 n - 1 位是否为 1，来判断数字 n 是否被使用过
            val bit = 1 shl (n - 1)
            if ((remainNumberBits and bit) > 0) {
                // 数字 n 可选
                // 我选 n 可以赢，或者我选 n 且后手不能赢
                if (n >= desiredTotal || !canWin(
                        desiredTotal - n,
                        (remainNumberBits and bit.inv()),
                        maxChoosableInteger
                    )
                ) {
                    return true.also { setMem(desiredTotal, remainNumberBits, it) }
                }
            }
        }

        return false.also { setMem(desiredTotal, remainNumberBits, it) }
    }

    fun canIWin(maxChoosableInteger: Int, desiredTotal: Int): Boolean {
        // 处理特殊情况，保证调用 canWin 时如果选完所有数字必然能满足 desiredTotal
        // 特殊情况1处理
        if (desiredTotal <= maxChoosableInteger) {
            return true
        }
        // 特殊情况2处理
        if (desiredTotal > ((maxChoosableInteger * (maxChoosableInteger + 1)) ushr 1)) {
            return false
        }

        // 一个 Int 共 32 bit，由于 1 <= maxChoosableInteger <= 20, 所以 1 个 bit 代表一位数的下标，足够用
        // 1代表还可以使用，0代表无法使用，maxChoosableInteger 个数都还可以使用可以用 2^(maxChoosableInteger) - 1 来标识
        return canWin(desiredTotal, (1 shl maxChoosableInteger) - 1, maxChoosableInteger)
    }
}

/// 464. 我能赢吗
class Solution464_20220522_1 {
    lateinit var mem: ShortArray
    fun canWin(desiredTotal: Int, remainNumberBits: Int, maxChoosableInteger: Int): Boolean {
        if (mem[remainNumberBits] != 0.toShort()) {
            return mem[remainNumberBits] == 1.toShort()
        }

        // remainNumberBits 不可能会等于 0，因为在调用前做了限制

        for (n in maxChoosableInteger downTo 1) {
            // n 即为数字 (从大到小遍历)
            // 通过判断 remainNumberBits 中第 n - 1 位是否为 1，来判断数字 n 是否被使用过
            val bit = 1 shl (n - 1)
            if ((remainNumberBits and bit) > 0) {
                // 数字 n 可选
                // 我选 n 可以赢，或者我选 n 且后手不能赢
                if (n >= desiredTotal || !canWin(
                        desiredTotal - n,
                        (remainNumberBits and bit.inv()),
                        maxChoosableInteger
                    )
                ) {
                    return true.also { mem[remainNumberBits] = 1.toShort() }
                }
            }
        }

        return false.also { mem[remainNumberBits] = 2.toShort() }
    }

    fun canIWin(maxChoosableInteger: Int, desiredTotal: Int): Boolean {
        // 处理特殊情况，保证调用 canWin 时如果选完所有数字必然能满足 desiredTotal
        // 特殊情况1处理
        if (desiredTotal <= maxChoosableInteger) {
            return true
        }
        // 特殊情况2处理
        if (desiredTotal > ((maxChoosableInteger * (maxChoosableInteger + 1)) ushr 1)) {
            return false
        }

        mem = ShortArray(1 shl maxChoosableInteger)
        // 一个 Int 共 32 bit，由于 1 <= maxChoosableInteger <= 20, 所以 1 个 bit 代表一位数的下标，足够用
        // 1代表还可以使用，0代表无法使用，maxChoosableInteger 个数都还可以使用可以用 2^(maxChoosableInteger) - 1 来标识
        return canWin(desiredTotal, (1 shl maxChoosableInteger) - 1, maxChoosableInteger)
    }
}


/// 944. 删列造序
class Solution944 {
    fun minDeletionSize(strs: Array<String>): Int {
        var count = 0
        for (c in 0 until strs[0].length) {
            for (r in 0 until strs.size - 1) {
                if (strs[r][c] > strs[r + 1][c]) {
                    count++
                    break
                }
            }
        }
        return count
    }
}

/// 704. 二分查找
class Solution704 {
    fun search(nums: IntArray, target: Int): Int {
        var l = 0
        var r = nums.size - 1

        while (l <= r) {
            var m = (l + r) ushr 1
            if (nums[m] == target) {
                return m
            } else if (nums[m] < target) {
                l = m + 1
            } else { // nums[m] > target
                r = m - 1
            }
        }

        return -1
    }

    fun search1(nums: IntArray, target: Int): Int {
        var l = 0
        var r = nums.size - 1

        while (l < r) {
            var m = (l + r) ushr 1
            if (nums[m] < target) {
                l = m + 1
            } else { // nums[m] >= target
                r = m
            }
        }

        if (l < nums.size && nums[l] == target) {
            return l
        }

        return -1
    }
}

/// 278. 第一个错误的版本
class Solution278 {
    fun isBadVersion(index: Int) = true
    fun firstBadVersion(n: Int): Int {
        var l = 0
        var r = n - 1

        while (l <= r) {
            val m = (l + r) ushr 1
            val isBad = isBadVersion(m + 1)
            if (isBad && (m == 0 || !isBadVersion(m))) {
                return m + 1
            } else if (isBad) {
                r = m - 1
            } else { // !isBad
                l = m + 1
            }
        }

        return -1
    }
}

/// 707. 设计链表
class MyLinkedList() {
    class Node(var `val`: Int, var next: Node? = null) {}

    var head: Node? = null

    fun getNodeAt(index: Int): Node? {
        if (index < 0) return null

        var i = 0
        var current = head
        while (i != index) {
            i++
            current = current?.next
        }
        return current
    }

    fun get(index: Int): Int {
        return getNodeAt(index)?.`val` ?: -1
    }

    fun addAtHead(`val`: Int) {
        val node = Node(`val`, head)
        head = node
    }

    fun addAtTail(`val`: Int) {
        var current = head
        if (current == null) {
            head = Node(`val`)
            return
        }

        while (current!!.next != null) {
            current = current.next
        }

        current.next = Node(`val`)
    }

    fun addAtIndex(index: Int, `val`: Int) {
        if (index <= 0) {
            addAtHead(`val`)
            return
        }

        var current = getNodeAt(index - 1)
        if (current != null) {
            var node = Node(`val`, current!!.next)
            current!!.next = node
        }
    }

    fun deleteAtIndex(index: Int) {
        if (index == 0) {
            head = head?.next
            return
        }

        var current = getNodeAt(index - 1)
        if (current != null) {
            current!!.next = current!!.next?.next
        }
    }

}

class MyLinkedList2() {
    class Node(var `val`: Int, var next: Node? = null, var prev: Node? = null) {}

    var head: Node? = null
    var tail: Node? = null

    fun getNode(index: Int): Node? {
        if (index < 0) return null

        var count = index
        var current = head
        while (count > 0) {
            current = current?.next
            count--
        }
        return current
    }

    fun get(index: Int): Int {
        return getNode(index)?.`val` ?: -1
    }

    fun addAtHead(`val`: Int) {
        val node = Node(`val`, head)
        if (head == null) {
            head = node
            tail = node
            return
        }

        head?.prev = node
        head = node
    }

    fun addAtTail(`val`: Int) {
        val node = Node(`val`)
        if (head == null) {
            head = node
            tail = node
            return
        }

        tail?.next = node
        node.prev = tail
        tail = node
    }

    fun addAtIndex(index: Int, `val`: Int) {
        val nodeAtIndexMinus1 = getNode(index - 1)
        if (nodeAtIndexMinus1 == null) {
            if (index == 0) {
                addAtHead(`val`)
            }
            return
        }

        if (nodeAtIndexMinus1 == tail) {
            addAtTail(`val`)
            return
        }

        val node = Node(`val`, nodeAtIndexMinus1.next, nodeAtIndexMinus1)
        nodeAtIndexMinus1.next = node
        node.next!!.prev = node
    }

    fun deleteAtIndex(index: Int) {
        val node = getNode(index)
        if (node == null) return

        val nodeIsHead = node == head
        val nodeIsTail = node == tail
        if (nodeIsHead && nodeIsTail) { // node 即是头结点也是尾节点
            head = null
            tail = null
        } else if (nodeIsHead) { // node 只是头结点
            head = node.next
            head!!.prev = null
        } else if (nodeIsTail) { // node 只是尾节点
            tail = node.prev
            tail!!.next = null
        } else {
            node.prev!!.next = node.next
            node.next!!.prev = node.prev
        }
    }

}

/// 35. 搜索插入位置
class Solution35 {
    fun searchInsert(nums: IntArray, target: Int): Int {
        if (nums[0] > target) {
            return 0
        }

        if (nums[nums.size - 1] < target) {
            return nums.size
        }

        // 此时，nums 中比然存在一个数大于或等于 target
        // 查找第一个大于或等于 target 的数
        var l = 0
        var r = nums.size - 1

        while (l <= r) {
            val mid = (l + r) ushr 1
            if (nums[mid] >= target && (mid == 0 || nums[mid - 1] < target)) {
                return mid
            } else if (nums[mid] >= target) {
                r = mid - 1
            } else { // nums[mid] < target
                l = mid + 1
            }
        }

        return -1
    }
}

/// 141. 环形链表
class Solution141 {
    fun hasCycle(head: ListNode?): Boolean {
        var walker = head
        var runner = head

        while (walker != null && runner?.next != null) {
            walker = walker?.next
            runner = runner!!.next?.next

            if (walker == runner) {
                return true
            }
        }

        return false
    }
}

/// 142. 环形链表 II
class Solution142 {
    fun detectCycle(head: ListNode?): ListNode? {
        var walker = head
        var runner = head
        var hasCircle = false
        while (walker != null && runner?.next != null) {
            walker = walker!!.next
            runner = runner!!.next?.next
            if (walker == runner) {
                hasCircle = true
                break
            }
        }

        if (!hasCircle) return null

        // 让 runner 减速，和 walker 保持一致速度
        walker = head
        while (walker != runner) {
            walker = walker!!.next
            runner = runner!!.next
        }
        return walker
    }
}

/// 202. 快乐数
class Solution202 {
    fun isHappy(n: Int): Boolean {
        var seen = mutableMapOf<Int, Boolean>()
        var sum = numberSum(n).also { seen[n] = true }
        while (sum != 1) {
            sum = numberSum(sum)
            if (!seen.containsKey(sum)) {
                seen[sum] = true
            } else {
                break
            }
        }
        return sum == 1
    }

    fun numberSum(n_: Int): Int {
        // return n.toString().map { it - '0'}.sumOf { it * it }
        var sum = 0
        var n = n_
        while (n > 0) {
            val v = (n % 10)
            sum += v * v
            n = n / 10
        }
        return sum
    }
}

/// 160. 相交链表
class Solution160 {
    fun getIntersectionNode(headA: ListNode?, headB: ListNode?): ListNode? {
        // 链A长度
        var a = headA
        var lengthA = 0
        while (a != null) {
            a = a?.next
            lengthA++
        }

        // 链B长度
        var b = headB
        var lengthB = 0
        while (b != null) {
            b = b?.next
            lengthB++
        }

        // 计算长度差
        var diff: Int
        a = headA
        b = headB
        // 假设后续 a 更长，b 更短
        if (lengthA >= lengthB) {
            diff = lengthA - lengthB
        } else {
            diff = lengthB - lengthA
            a = b.also { b = a }
        }

        // 优先移动长链表
        // 然后一起移动链表，检测是否找到相交点
        while (a != null && b != null) {
            if (a == b) {
                return a
            }
            a = a?.next
            if (diff > 0) {
                diff--
                continue
            }
            b = b?.next
        }

        // 未找到相交点的情况
        return null
    }
}

// 19. 删除链表的倒数第 N 个结点
class Solution19 {
    fun removeNthFromEnd(head: ListNode?, n: Int): ListNode? {
        var prehead = ListNode(0, head)
        var count = n
        var current = prehead
        var target = prehead

        while (current.next != null) {
            current = current.next!!
            if (count > 0) {
                count--
                continue
            }
            target = target.next!!
        }
        target!!.next = target!!.next?.next

        return prehead.next
    }

    fun removeNthFromEnd1(head: ListNode?, n: Int): ListNode? {
        var prehead = ListNode(0)
        prehead.next = head

        // 要删除倒数第 n 个节点，需要找到倒数第 n+1 个节点。

        var target: ListNode? = prehead
        var current: ListNode? = prehead
        var i = 0
        while (current?.next != null) {
            current = current.next!!
            if (i < n) { // 保证 current 先走 n 步，之后 target 和 current 同步走，直到 current 到达最后一个节点，此时 target 即为倒数第 n+1 个节点
                i++
                continue
            }
            target = target!!.next
        }
        target!!.next = target!!.next?.next

        return prehead.next
    }
}

/// 206. 反转链表
class Solution206 {
    fun reverseList(head: ListNode?): ListNode? {
        var pre: ListNode? = null
        var cur = head

        while (cur != null) {
            val curNext = cur?.next
            cur.next = pre
            // 为下次迭代准备
            pre = cur
            cur = curNext
        }

        return pre
    }
}

/// 203. 移除链表元素
class Solution203 {
    fun removeElements(head: ListNode?, value: Int): ListNode? {
        var prehead = ListNode(0, head)
        var cur: ListNode? = prehead
        while (cur?.next != null) {
            if (cur!!.next?.`val` == value) {
                cur!!.next = cur!!.next?.next
            } else {
                cur = cur.next
            }
        }
        return prehead.next
    }
}

/// 328. 奇偶链表
class Solution328 {
    fun oddEvenList0(head: ListNode?): ListNode? {
        if (head == null) return null

        var odd: ListNode = head!!
        var even: ListNode? = head!!.next
        var evenHead: ListNode? = even

        while (even != null && even!!.next != null) {
            odd.next = even.next
            even.next = even.next?.next

            odd = odd.next!!
            even = even.next
        }
        odd.next = evenHead

        return head
    }

    fun oddEvenList(head: ListNode?): ListNode? {
        var even: ListNode? = ListNode(0)
        var odd: ListNode? = ListNode(0)
        var evenHead = even
        var oddHead = odd

        var current = head
        while (current != null) {
            val currentNext = current.next
            val currentNextNext = currentNext?.next

            current.next = null
            currentNext?.next = null

            odd?.next = current
            odd = current

            if (currentNext != null) {
                even?.next = currentNext
                even = currentNext
                current = currentNextNext
            } else {
                current = null // current.next
            }
        }

        odd?.next = evenHead!!.next

        return oddHead?.next
    }
}

/// 234. 回文链表
class Solution234 {
    fun reverseList(head: ListNode?): ListNode? {
        var pre: ListNode? = null
        var cur = head
        while (cur != null) {
            val curNext = cur.next
            cur.next = pre
            pre = cur
            cur = curNext
        }

        return pre
    }

    fun isPalindrome(head: ListNode?): Boolean {
        // 计算链表长度
        var length = 0
        var current = head
        while (current != null) {
            length++
            current = current.next
        }

        if (length <= 1) return true

        // 找到中间节点
        length = length / 2 - 1
        var middle = head!!
        while (length > 0) {
            length--
            middle = middle.next!!
        }

        // 拆成前后两条链 head 代表前链，post 代表后链
        var post = middle.next!!
        middle.next = null

        // 后链翻转
        post = reverseList(post)!!

        // 前链和翻转后的后链对比 记录结果
        var isPalindrome = true
        var currentPost: ListNode? = post
        current = head
        while (current != null && currentPost != null) {
            if (current!!.`val` != currentPost!!.`val`) {
                isPalindrome = false
                break
            }
            current = current?.next
            currentPost = currentPost?.next
        }

        // 后链翻转（非必须）
        post = reverseList(post)!!
        // 前量接回后链（非必须）
        middle.next = post

        // 返回结果
        return isPalindrome
    }

    /// 快慢指针找中间节点
    fun isPalindrome1(head: ListNode?): Boolean {
        // 找到中间节点
        var current = head
        var middle = head
        while (current?.next?.next != null) {
            current = current!!.next!!.next
            middle = middle!!.next
        }

        // 拆成前后两条链 head 代表前链，post 代表后链
        var post = middle?.next
        middle?.next = null

        // 后链翻转
        post = reverseList(post)

        // 前链和翻转后的后链对比 记录结果
        var isPalindrome = true
        var currentPost: ListNode? = post
        current = head
        while (current != null && currentPost != null) {
            if (current!!.`val` != currentPost!!.`val`) {
                isPalindrome = false
                break
            }
            current = current?.next
            currentPost = currentPost?.next
        }

        // 后链翻转（非必须）
        post = reverseList(post)
        // 前量接回后链（非必须）
        middle?.next = post

        // 返回结果
        return isPalindrome
    }
}

/// 21. 合并两个有序链表
class Solution21 {
    fun mergeTwoLists(list1: ListNode?, list2: ListNode?): ListNode? {
        var l1 = list1
        var l2 = list2
        if (l1 == null) return l2
        if (l2 == null) return l1

        var prehead = ListNode(0)
        var current = prehead
        while (l1 != null && l2 != null) {
            if (l1!!.`val` > l2!!.`val`) {
                current.next = l2!!
                l2 = l2!!.next
            } else {
                current.next = l1!!
                l1 = l1!!.next
            }
            current = current.next!!
        }
        current.next = l1 ?: l2

        return prehead.next
    }
}

/// 430. 扁平化多级双向链表
class Solution430 {
    class Node(var `val`: Int) {
        var prev: Node? = null
        var next: Node? = null
        var child: Node? = null
    }

    fun flatten(root: Node?): Node? {
        var flattenPrehead = Node(0)
        var flatten: Node? = flattenPrehead
        var current = root
        var stack = mutableListOf<Node>()

        while (current != null) {
            flatten?.next = current
            current.prev = flatten

            if (current.child != null) {
                if (current?.next != null) {
                    stack.add(current.next!!)
                    // current.next!!.prev = null
                    // current.next = null
                }

                val currentChild = current.child
                current.child = null
                current = currentChild
            } else {
                val currentNext = current.next
                // currentNext?.prev = null
                // current.next = null
                current = currentNext
                if (current == null && stack.size > 0) {
                    current = stack.removeAt(stack.size - 1)
                }
            }

            flatten = flatten?.next
        }

        var result = flattenPrehead.next
        result?.prev = null

        return result
    }
}

/// 138. 复制带随机指针的链表
class Solution138 {
    class Node(var `val`: Int) {
        var next: Node? = null
        var random: Node? = null
    }

    fun copyNode(node: Node?, mem: MutableMap<Int, Node>): Node? {
        if (node == null) return null
        val copy = Node(node.`val`)
        copy.next = copyNode(node.next, mem)
        copy.random = node.random
        mem[node.hashCode()] = copy
        return copy
    }

    fun copyRandomList(node: Node?): Node? {
        var mem = mutableMapOf<Int, Node>()
        var copy = copyNode(node, mem)
        var current = copy
        while (current != null) {
            current.random = mem[current.random.hashCode()]
            current = current.next
        }
        return copy
    }

    /////

    var cachedNode: MutableMap<Node, Node> = HashMap()
    fun copyRandomList1(head: Node?): Node? {
        if (head == null) {
            return null
        }
        if (!cachedNode.containsKey(head)) {
            val headNew = Node(head.`val`)
            cachedNode[head] = headNew
            headNew.next = copyRandomList1(head.next)
            headNew.random = copyRandomList1(head.random)
        }
        return cachedNode[head]
    }
}

/// 61. 旋转链表
class Solution61 {
    // 确定链表长度 L
    // k = k % L
    // 找到倒数第 k + 1 个节点 tail
    // 断开倒数第 k+1 个节点和倒数第 k 个节点，新链表表头为原链表倒数第 k 个节点
    // 将新链表接到原链表前面
    fun rotateRight(head: ListNode?, k: Int): ListNode? {
        // 确定链表长度 L
        var length = 0
        var current = head
        while (current != null) {
            length++
            current = current.next
        }

        if (length <= 1) return head

        // k = k % L
        var theK = k % length

        if (theK == 0) return head

        // 找到倒数第 k + 1 个节点 tail
        current = head
        var theTail = head
        while (current?.next != null) {
            current = current!!.next
            if (theK > 0) {
                theK--
                continue
            }
            theTail = theTail?.next
        }

        // 断开倒数第 k+1 个节点和倒数第 k 个节点，新链表表头为原链表倒数第 k 个节点
        var theHead = theTail!!.next
        theTail.next = null
        // 将新链表接到原链表前面
        current?.next = head

        return theHead
    }
}

/// 977. 有序数组的平方
class Solution977 {
    fun binarySearchMaxNegative(nums: IntArray): Int {
        var l = 0
        var r = nums.size - 1

        while (l <= r) {
            val mid = (l + r) ushr 1
            if (nums[mid] < 0 && (mid == nums.size - 1 || nums[mid + 1] >= 0)) {
                return mid
            } else if (nums[mid] < 0) {
                l = mid + 1
            } else { // nums[mid] >= 0
                r = mid - 1
            }
        }
        return -1
    }

    fun sortedSquares1(nums: IntArray): IntArray {
        // 二分查找最大的那个负数
        var negativeIndex = binarySearchMaxNegative(nums)
        // 两个下标（负数部分，正数部分）向两个方向遍历
        var nonnegativeIndex = negativeIndex + 1

        var result = IntArray(nums.size)
        var i = 0

        while (negativeIndex >= 0 || nonnegativeIndex < nums.size) {
            val negative = nums.getOrElse(negativeIndex) { Int.MIN_VALUE }
            val nonnegative = nums.getOrElse(nonnegativeIndex) { Int.MAX_VALUE }

            if (negative + nonnegative < 0) {
                // nonnegative 绝对值更小，所以 nonnegative^2 更小
                result[i] = nonnegative * nonnegative
                nonnegativeIndex++
            } else {
                result[i] = negative * negative
                negativeIndex--
            }
            i++
        }

        return result
    }

    fun sortedSquares(nums: IntArray): IntArray {
        var result = IntArray(nums.size)

        var low = 0
        var high = nums.size - 1
        var i = high

        while (low <= high) {
            val lowValue = if (nums[low] < 0) -nums[low] else nums[low]
            val highValue = if (nums[high] < 0) -nums[high] else nums[high]

            if (lowValue < highValue) {
                result[i] = highValue * highValue
                high--
            } else {
                result[i] = lowValue * lowValue
                low++
            }
            i--
        }

        return result
    }
}

// 求两数的最大公约数
// 欧几里得算法
class Utils {
    companion object {
        // 计算最大公约数（欧几里得算法）
        fun gcd(a: Int, b: Int): Int {
            if (b == 0) {
                return a
            }
            return gcd(b, a % b)
        }

        // lcm(a,b) = a*b / gcd(a,b)
        fun lcm(a: Int, b: Int): Int {
            return a * b / gcd(a, b)
        }

        // 三角形面积 海伦公式 √[p(p-a)(p-b)(p-c)], p=(a+b+c)/2.
        fun triangleArea(points: Array<IntArray>): Double {
            var x = points[0][0] - points[1][0].toDouble()
            var y = points[0][1] - points[1][1].toDouble()

            var a = Math.sqrt(x * x + y * y)

            x = points[0][0] - points[2][0].toDouble()
            y = points[0][1] - points[2][1].toDouble()
            var b = Math.sqrt(x * x + y * y)

            x = points[1][0] - points[2][0].toDouble()
            y = points[1][1] - points[2][1].toDouble()
            var c = Math.sqrt(x * x + y * y)

            val p = (a + b + c) / 2
            return Math.sqrt(p * (p - a) * (p - b) * (p - c))
        }

        fun distance(points: Array<IntArray>): Double {
            var x = points[0][0] - points[1][0].toDouble()
            var y = points[0][1] - points[1][1].toDouble()
            return Math.sqrt(x * x + y * y)
        }
    }
}

/// 189. 轮转数组
class Solution189 {
    fun rotate(nums: IntArray, k: Int): Unit {
        if (nums.size <= 1) return

        var theK = k % nums.size
        for (i in 0 until theK) {
            var tmp = nums[nums.size - 1]
            for (j in nums.size - 1 downTo 1) {
                nums[j] = nums[j - 1]
            }
            nums[0] = tmp
        }
    }

    fun rotate1(nums: IntArray, k: Int): Unit {
        val n = nums.size
        var k = k % n
        var count = gcd(k, n)
        for (start in 0 until count) {
            var current = start
            var prev = nums[start]
            do {
                var next = (current + k) % n
                var tmp = nums[next]
                nums[next] = prev
                prev = tmp
                current = next
            } while (start != current)
        }
    }

    fun gcd(a: Int, b: Int): Int {
        if (b == 0) return a
        return gcd(b, a % b)
    }

    /// 翻转法

    fun reverse(nums: IntArray, start: Int, end: Int) {
        var l = start
        var r = end
        while (l <= r) {
            nums[l] = nums[r].also { nums[r] = nums[l] }
            l++
            r--
        }
    }

    fun rotate2(nums: IntArray, k: Int): Unit {
        var n = nums.size
        var k = k % n

        reverse(nums, 0, n - 1)
        reverse(nums, 0, k - 1)
        reverse(nums, k, n - 1)
    }
}

/// 面试题 01.05. 一次编辑
class Solution0105 {
    fun oneEditAway(first: String, second: String): Boolean {
        // 长度相差大于等于2，则无法通过1次编辑就让两个字符串相等
        if (first.length - second.length >= 2 || first.length - second.length <= -2) return false

        // 长度相等
        if (first.length == second.length) {
            var count = 0
            for (i in first.indices) {
                if (first[i] != second[i]) count++
                if (count >= 2) break
            }
            return count <= 1
        }

        // 长度差1
        var longer = first
        var shorter = second
        if (shorter.length > longer.length) {
            longer = second
            shorter = first
        }

        for (i in longer.indices) {
            if (longer.removeRange(i until i + 1) == shorter) {
                return true
            }
        }
        return false
    }
}

/// 821. 字符的最短距离
class Solution821 {
    fun shortestToChar(s: String, c: Char): IntArray {
        var indices = mutableListOf<Int>()
        for (i in s.indices) {
            if (s[i] == c) {
                indices.add(i)
            }
        }

        var bounds = mutableListOf<Int>()
        for (i in 1 until indices.size) {
            bounds.add((indices[i] + indices[i - 1] + 1) ushr 1)
        }
        bounds.add(s.length)

        var result = IntArray(s.length)
        var i = 0
        for (b in 0 until bounds.size) {
            val bound = bounds[b]
            val index = indices[b]
            while (i < bound) {
                result[i] = kotlin.math.abs(i - index)
                i++
            }
        }

        return result
    }
}

/// 283. 移动零
class Solution283 {
    fun moveZeroes(nums: IntArray): Unit {
        var cur = 0
        var i = 0
        while (cur < nums.size) {
            if (nums[i] != 0) {
                i++
                cur = i + 1
                continue
            }
            if (nums[cur] != 0 && cur != i) {
                nums[i] = nums[cur]
                nums[cur] = 0
                i++
            }
            cur++
        }
    }
}

/// 167. 两数之和 II - 输入有序数组
class Solution167 {
    fun twoSum(numbers: IntArray, target: Int): IntArray {
        var low = 0
        var high = numbers.size - 1
        while (low < high) {
            // 『第一步』：判断当前最大的数是否不可能为目标，去除不可能的
            while (numbers[high] + numbers[low] > target) {
                high--
            }
            // 『第二步』：判断当前最小的数是否不可能为目标，去除不可能的
            while (numbers[high] + numbers[low] < target) {
                low++
            }
            // 『第三步』：此时，numbers[high] + numbers[low] >= target，相等时直接返回，否则继续循环回到 『第一步』
            if (numbers[high] + numbers[low] == target) {
                return intArrayOf(low + 1, high + 1)
            }
        }
        return intArrayOf()
    }

    fun twoSum2(numbers: IntArray, target: Int): IntArray {
        var low = 0
        var high = numbers.size - 1
        while (low < high) {
            // 『第一步』：判断当前最大的数是否不可能为目标，去除不可能的
            var targetIndex =
                BinarySearch.indexOfLessThanOrEqual(numbers, target - numbers[low], low, high)
            if (numbers[targetIndex] + numbers[low] == target) {
                return intArrayOf(low + 1, targetIndex + 1)
            } else {
                high = targetIndex
            }

            // 『第二步』：判断当前最小的数是否不可能为目标，去除不可能的
            targetIndex =
                BinarySearch.indexOfGreaterThanOrEqual(numbers, target - numbers[high], low, high)
            if (numbers[targetIndex] + numbers[low] == target) {
                return intArrayOf(low + 1, targetIndex + 1)
            } else {
                low = targetIndex
            }
        }
        return intArrayOf()
    }

    fun twoSum1(numbers: IntArray, target: Int): IntArray {
        var low = 0
        var high = numbers.size - 1
        while (low < high) {
            val sum = numbers[high] + numbers[low]
            if (sum > target) {
                high--
            } else if (sum < target) {
                low++
            } else { // (numbers[high] + numbers[low] == target)
                return intArrayOf(low + 1, high + 1)
            }
        }
        return intArrayOf()
    }
}

/// 691. 贴纸拼词
class Solution691 {
    fun minStickers(stickers: Array<String>, target: String): Int {
        val nextTargetCache = mutableMapOf<String, MutableMap<String, String>>()
        fun nextTarget(target: String, sticker: String): String {

            fun getCache(target: String, sticker: String): String? {
                if (nextTargetCache.containsKey(target)) {
                    if (nextTargetCache[target]!!.containsKey(sticker)) {
                        return nextTargetCache[target]!![sticker]!!
                    }
                }
                return null
            }

            fun setCache(target: String, sticker: String, nextTarget: String) {
                if (!nextTargetCache.containsKey(target)) {
                    nextTargetCache[target] = mutableMapOf<String, String>()
                }
                nextTargetCache[target]!![sticker] = nextTarget
            }

            var result = getCache(target, sticker)
            if (result != null) {
                return result
            }

            var next = StringBuffer("")
            var stickerChars = sticker.map { it }.toMutableList()
            target.forEach {
                val index = stickerChars.indexOf(it)
                if (index != -1) { // 找到了，不记录，并从 stickerChars 中删除
                    stickerChars.removeAt(index)
                } else {
                    next.append(it)
                }
            }

            val nextTargetString = next.toString()

            setCache(target, sticker, nextTargetString)

            return nextTargetString
        }

        // minTargetCounts[target] 代表拼成目标 target 字符串需要使用的最小贴纸数量
        var minCountCache = mutableMapOf<String, Int>()

        val theStickers = stickers.map {
            it.filter { c -> target.contains(c) }
        }.filter { it.isNotEmpty() }

        fun searchMinTargetCount(target: String): Int {
            if (minCountCache.containsKey(target)) {
                return minCountCache[target]!!
            }

            if (target.isEmpty()) {
                return 0.also { minCountCache[target] = it }
            }

            var targetMinCount = Int.MAX_VALUE
            for (sticker in theStickers) {
                val nextTarget = nextTarget(target, sticker)
                if (target == nextTarget) {
                    // target 和 sticker 没有交集，sticker 对构造当前 target 没有用
                    continue
                }

                val nextTargetMinCount = searchMinTargetCount(nextTarget)
                if (nextTargetMinCount == Int.MAX_VALUE) {
                    continue
                }

                targetMinCount = minOf(targetMinCount, 1 + nextTargetMinCount)
            }

            return targetMinCount.also { minCountCache[target] = it }
        }

        val theTargetMinCount = searchMinTargetCount(target)
        if (theTargetMinCount == Int.MAX_VALUE) {
            return -1
        }
        return theTargetMinCount
    }

    ///////////////////////////////

    fun combination(string: String, bitCount: Int): List<List<Char>> {
        if (bitCount == 1) {
            return string.map { listOf(it) }
        }

        var resultList = mutableListOf<List<Char>>()
        for (i in 0 until string.length) {
            combination(string.substring(i + 1), bitCount - 1).forEach {
                var list = it.toMutableList()
                list.add(0, string[i])
                resultList.add(list)
            }
        }
        return resultList
    }

    fun bitCombination(bitsMap: Map<Char, MutableList<Int>>, chars: List<Char>): List<Int> {
        if (chars.size == 0) return listOf<Int>()
        if (chars.size == 1) {
            var result = bitsMap[chars.first()] ?: listOf<Int>()
            return result
        }

        var otherBitValues = bitCombination(bitsMap, chars.subList(1, chars.size))

        val bitValues = mutableListOf<Int>()
        val bitList = bitsMap[chars.first()]!!
        for (bit in bitList) {
            otherBitValues.forEach {
                bitValues.add(bit or it)
            }
        }

        return bitValues
    }

    fun minStickers1(stickers: Array<String>, target: String): Int {
        val targetValue = (Math.pow(2.0, target.length.toDouble()) - 1).toInt()

        var stickerValues = mutableListOf<Int>()

        var stickers = stickers.map { s -> s.filter { target.indexOf(it) != -1 } }
        for (sticker in stickers) {
            var stickerValueBits = mutableMapOf<Char, MutableList<Int>>()
            target.forEachIndexed { index, c ->
                val found = sticker.indexOf(c)
                if (found != -1) {
                    if (!stickerValueBits.containsKey(c)) {
                        stickerValueBits[c] = mutableListOf()
                    }
                    stickerValueBits[c]!!.add(1 shl index)
                }
            }
            for (count in 1 until sticker.length + 1) {
                val charsList = combination(sticker, count)
                charsList.forEach {
                    stickerValues.addAll(bitCombination(stickerValueBits, it))
                }
            }
            stickerValues = stickerValues
        }

        var mem = mutableMapOf<Int, Int>()
        fun minimumCount(state: Int): Int {
            if (mem.containsKey(state)) {
                return mem[state]!!
            }

            if (state == 0) {
                return 0.also { mem[state] = it }
            }

            var min = Int.MAX_VALUE
            for (stickerValue in stickerValues) {
                val newState = state and stickerValue.inv()
                if (newState == state) {
                    // 选择当前的 stickerValue 对应的 sticker 无法改变当前状态，即操作无效
                    continue
                }
                var newStateMinCount = minimumCount(newState)
                if (newStateMinCount == Int.MAX_VALUE) {
                    // 子状态对应的字符串无法被构造
                    continue
                }
                min = minOf(min, 1 + newStateMinCount)
            }
            return min.also { mem[state] = it }
        }

        val result = minimumCount(targetValue)
        if (result == Int.MAX_VALUE) return -1
        return result
    }

    ////////////

    fun minStickers2(stickers: Array<String>, target: String): Int {
        val targetValue = (Math.pow(2.0, target.length.toDouble()) - 1).toInt()

        var stickerValues = mutableListOf<Int>()

        var stickers =
            stickers.map { s -> s.filter { target.indexOf(it) != -1 } }.filter { it.isNotEmpty() }
        for (sticker in stickers) {
            var stickerValueBits = mutableMapOf<Char, MutableList<Int>>()
            target.forEachIndexed { index, c ->
                val found = sticker.indexOf(c)
                if (found != -1) {
                    if (!stickerValueBits.containsKey(c)) {
                        stickerValueBits[c] = mutableListOf()
                    }
                    stickerValueBits[c]!!.add(1 shl index)
                }
            }
            val charsList = combination(sticker, sticker.length)
            charsList.forEach {
                stickerValues.addAll(bitCombination(stickerValueBits, it))
            }
            stickerValues = stickerValues
        }

        // minCountCache[targetValue] 代表拼成目标 target 字符串需要使用的最小贴纸数量
        var minCountCache = mutableMapOf<Int, Int>()
        minCountCache[0] = 0

        for (currentTarget in 1 until targetValue + 1) {
            var minCount = Int.MAX_VALUE
            for (stickerValue in stickerValues) {
                val nextTarget = currentTarget and stickerValue.inv()
                if (nextTarget == currentTarget) {
                    continue
                }
                val nextTargetMinCount = minCountCache[nextTarget]!!
                if (nextTargetMinCount == Int.MAX_VALUE) {
                    continue
                }
                minCount = minOf(minCount, 1 + nextTargetMinCount)
                if (minCount == 1) {
                    break
                }
            }
            minCountCache[currentTarget] = minCount
        }

        val result = minCountCache[targetValue]!!
        if (result == Int.MAX_VALUE) return -1
        return result
    }
}

/// 11. 盛最多水的容器
class Solution11 {
    fun maxArea(heights: IntArray): Int {
        fun area(low: Int, high: Int): Int = minOf(heights[low], heights[high]) * (high - low)

        var l = 0
        var r = heights.size - 1

        var max = area(l, r)
        while (l < r) {
            var leftArea = area(l + 1, r)
            if (leftArea > max) {
                max = leftArea
                l += 1
                continue
            }

            var rightArea = area(l, r - 1)
            if (rightArea > max) {
                max = rightArea
                r -= 1
                continue
            }

            if (heights[l] < heights[r]) {
                l += 1
            } else {
                r -= 1
            }
        }

        return max
    }

    fun maxAreaBetter(heights: IntArray): Int {
        fun area(low: Int, high: Int): Int = minOf(heights[low], heights[high]) * (high - low)

        var l = 0
        var r = heights.size - 1

        var max = 0
        while (l < r) {
            var tmp = area(l, r)
            if (tmp > max) {
                max = tmp
            }
            if (heights[l] < heights[r]) {
                l += 1
            } else {
                r -= 1
            }
        }

        return max
    }
}

class Solution11_20220529 {
    fun maxArea(height: IntArray): Int {
        var left = 0
        var right = height.size - 1

        var max = 0
        while (left < right) {
            if (height[left] < height[right]) {
                max = maxOf(max, (right - left) * height[left])
                left++
            } else {
                max = maxOf(max, (right - left) * height[right])
                right--
            }
        }

        return max
    }
}

/// 344. 反转字符串
class Solution344 {
    fun reverseString(s: CharArray): Unit {
        var i = 0
        while (i < s.size / 2) {
            var j = s.size - 1 - i
            s[i] = s[j].also { s[j] = s[i] }
            i++
        }
    }
}

/// 557. 反转字符串中的单词 III
class Solution557 {
    fun reverseWords(str: String): String {
        var start = 0
        var current = 0
        var s = str.map { it }.toMutableList()

        while (current < s.size + 1) {
            val currentIsSpace = (current == s.size || s[current] == ' ')
            if (currentIsSpace) {
                // 翻转 start ~ current - 1
                var end = current - 1
                while (start < end) {
                    s[start] = s[end].also { s[end] = s[start] }
                    start++
                    end--
                }
                // 恢复 start，为下个单词准备
                start = current + 1
            }
            current++
        }

        return s.joinToString("")
    }

    fun reverseWords1(str: String): String {
        var start = 0
        var current = 0
        var s = StringBuffer("")

        while (current < str.length + 1) {
            val currentIsSpace = (current == str.length || str[current] == ' ')
            if (currentIsSpace) {
                var end = current - 1
                while (start <= end) {
                    s.append(str[end])
                    end--
                }
                s.append(if (current == str.length) "" else " ")
                // 恢复 start，为下个单词准备
                start = current + 1
            }
            current++
        }

        return s.toString()
    }
}

/// 876. 链表的中间结点
class Solution876 {
    /// 快慢指针寻找链表右中位数
    fun middleNode(head: ListNode?): ListNode? {
        var walker = head
        var runner = head
        while (runner?.next != null) {
            runner = runner!!.next?.next
            walker = walker!!.next
        }
        return walker
    }

    /// 朴素方法
    fun middleNode1(head: ListNode?): ListNode? {
        var current = head
        var length = 0
        while (current != null) {
            length++
            current = current.next
        }

        length = (length ushr 1) - 1

        current = head
        while (length >= 0) {
            current = current?.next
            length--
        }

        return current
    }
}

/// 812. 最大三角形面积
class Solution812 {
    fun largestTriangleArea(points: Array<IntArray>): Double {
        fun distance(i: Int, j: Int): Double {
            val x = points[i][0] - points[j][0]
            val y = points[i][1] - points[j][1]
            val result = Math.sqrt((x * x + y * y).toDouble())
            return result
        }

        /* 三阶行列式，主对角线（左上到右下） - 副对角线（左下到右上）
         x1 y1 1
         x2 y2 1
         x3 y3 1
               x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2
         */
        // abs(x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2) / 2
        fun triangleArea1(i: Int, j: Int, k: Int): Double {
            return Math.abs(points[i][0] * points[j][1] + points[j][0] * points[k][1] + points[k][0] * points[i][1] - points[i][0] * points[k][1] - points[j][0] * points[i][1] - points[k][0] * points[j][1].toDouble()) / 2
        }

        // 三角形面积 海伦公式 √[p(p-a)(p-b)(p-c)], p=(a+b+c)/2.
        fun triangleArea(i: Int, j: Int, k: Int): Double {
            var a = distance(i, j)
            var b = distance(i, k)
            var c = distance(j, k)
            val p = (a + b + c) / 2
            var sq = p * (p - a) * (p - b) * (p - c)
            if (sq < 0) {
                sq = -sq
            }
            val result = Math.sqrt(sq)
            if (result.isNaN()) {
                return result
            }
            return result
        }

        var max = 0.0
        for (i in 0 until points.size - 2) {
            for (j in i + 1 until points.size - 1) {
                for (k in j + 1 until points.size) {
                    max = maxOf(max, triangleArea(i, j, k))
                }
            }
        }

        return max
    }
}

/// 42. 接雨水
class Solution42 {
    fun trap1(walls: IntArray): Int {

        fun calSumWithWater(ascendWallIndices: List<Int>): Int {
            if (ascendWallIndices.size == 1) {
                return walls[ascendWallIndices[0]]
            }

            var sum = 0
            for (i in 0 until ascendWallIndices.size - 1) {
                val lowWallIndex = ascendWallIndices[i]
                val highWallIndex = ascendWallIndices[i + 1]
                val lowWall = walls[lowWallIndex]
                val highWall = walls[highWallIndex]
                sum += (highWall - lowWall) + (Math.abs(lowWallIndex - highWallIndex) + 1) * lowWall
                if (i > 0) {
                    sum -= lowWall
                }
            }
            return sum
        }

        var sum = 0 // 墙体高度总和
        var ascendWallIndices = mutableListOf<Int>()

        var current = 0
        var i = 0
        while (i < walls.size) {
            if (walls[i] >= current) {
                ascendWallIndices.add(i)
                current = walls[i]
            }
            sum += walls[i]
            i++
        }

        var bigSum = calSumWithWater(ascendWallIndices)

        var left = ascendWallIndices.last()
        current = 0
        i = walls.size - 1
        ascendWallIndices = mutableListOf<Int>()
        while (i >= left) {
            if (walls[i] >= current) {
                ascendWallIndices.add(i)
                current = walls[i]
            }
            i--
        }

        bigSum += calSumWithWater(ascendWallIndices)

        // 减掉多加的那堵最高的墙
        bigSum -= walls[ascendWallIndices.last()]

        return bigSum - sum
    }

    fun trap2(walls: IntArray): Int {
        var sumHeight = 0

        var prevIndex = -1
        var currentHeight = 0
        var originSumHeight = 0

        var i = 0
        while (i < walls.size) {
            if (walls[i] >= currentHeight) {
                if (i - prevIndex > 1) {
                    // 计算
                    val sum = currentHeight * (i - prevIndex - 1)
                    sumHeight += (sum - originSumHeight)
                }
                currentHeight = walls[i]
                originSumHeight = 0
                prevIndex = i
            } else {
                originSumHeight += walls[i]
            }
            i++
        }

        val theHighestWallIndex = prevIndex
        prevIndex = walls.size
        currentHeight = 0
        originSumHeight = 0

        i = walls.size - 1
        while (i >= theHighestWallIndex) {
            if (walls[i] >= currentHeight) {
                if (prevIndex - i > 1) {
                    // 计算
                    val sum = currentHeight * (prevIndex - i - 1)
                    sumHeight += (sum - originSumHeight)
                }
                currentHeight = walls[i]
                originSumHeight = 0
                prevIndex = i
            } else {
                originSumHeight += walls[i]
            }
            i--
        }

        return sumHeight
    }

    fun trap(heights: IntArray): Int {
        var ans = 0

        var left = 0
        var leftMax = 0

        var right = heights.size - 1
        var rightMax = 0
        while (left < right) {
            if (leftMax < heights[left]) {
                leftMax = heights[left]
            }
            if (rightMax < heights[right]) {
                rightMax = heights[right]
            }
            if (heights[left] < heights[right]) {
                ans += leftMax - heights[left]
                left++
            } else {
                ans += rightMax - heights[right]
                --right
            }
        }

        return ans
    }
}

class Solution42_20220524 {
    fun trap(walls: IntArray): Int {
        var highestWall = 0
        var highestWallIndex = -1
        var waterTotal = 0
        var wallTotal = 0

        for (i in walls.indices) {
            if (walls[i] >= highestWall) {
                waterTotal += (i - highestWallIndex - 1) * highestWall
                waterTotal -= wallTotal

                highestWallIndex = i
                highestWall = walls[i]
                wallTotal = 0
            } else {
                wallTotal += walls[i]
            }
        }

        val theHighestWallIndex = highestWallIndex
        highestWall = 0
        highestWallIndex = walls.size
        wallTotal = 0
        for (i in walls.size - 1 downTo theHighestWallIndex) {
            if (walls[i] >= highestWall) {
                waterTotal += (highestWallIndex - i - 1) * highestWall
                waterTotal -= wallTotal

                highestWallIndex = i
                highestWall = walls[i]
                wallTotal = 0
            } else {
                wallTotal += walls[i]
            }
        }

        return waterTotal
    }

    /// 记录左右边界最高的墙, 让低的一方先移动, 因为水即使漫过低的边界也不可能漫过高的边界.
    /// 每次迭代统计一次可接水量: leftMax - leftCurrent, rightMax - rightCurrent
    fun trap_twoPointers(walls: IntArray): Int {
        var left = 0
        var right = walls.size - 1
        var leftMax = 0
        var rightMax = 0

        var total = 0

        while (left < right) {
            val wallLeft = walls[left]
            val wallRight = walls[right]
            if (wallLeft > leftMax) leftMax = wallLeft
            if (wallRight > rightMax) rightMax = wallRight
            if (wallLeft < wallRight) {
                total += leftMax - wallLeft
                left++
            } else {
                total += rightMax - wallRight
                right--
            }
        }

        return total
    }
}

/// 198. 打家劫舍
class Solution198 {
    fun rob(nums: IntArray): Int {
        if (nums.size == 0) return 0
        if (nums.size == 1) return nums[0]

        // dp[i] 代表在第 0~i 这 i+1 栋房屋可以获得的最大金额
        // dp[0] = nums[0]
        val n = nums.size
        var dp = IntArray(n)
        dp[0] = nums[0]
        dp[1] = maxOf(nums[0], nums[1])

        for (i in 2 until n) {
            dp[i] = maxOf(
                dp[i - 2] + nums[i],
                dp[i - 1]
            )
        }

        return dp[n - 1]
    }

    fun rob1(nums: IntArray): Int {
        // dp[i][j] 代表小偷从房屋 i 到 j 总共能够获取的最大金额
        /*  dp[i][j] = maxOf(
                nums[i] + dp[i + 2][j],
                nums[i + 1] + dp[i + 3][j]
                ...
                dp[i][i + k - 2] + nums[i + k] + dp[i + k + 2][j]
            )
            i <= j

            显然，dp[i][i] = nums[i]
        */
        var dp = Array(nums.size) { IntArray(nums.size) }

        for (i in dp.indices) {
            dp[i][i] = nums[i]
        }

        for (i in nums.size - 2 downTo 0) {
            for (j in i + 1 until nums.size) {
                var max = 0
                for (k in 0 until j - i + 1) {
                    var sum = if (i + k - 2 >= 0) dp[i][i + k - 2] else 0
                    sum += nums[i + k]
                    sum += if (i + k + 2 < nums.size) dp[i + k + 2][j] else 0
                    max = maxOf(max, sum)
                }
                dp[i][j] = max
            }
        }

        return dp[0][nums.size - 1]
    }
}

class Solution198_20220523 {
    fun rob(nums: IntArray): Int {
        // amount[i] 代表偷第0到第i家的最大金额，所以所求即为 amount[nums.size - 1]
        // amount[i] = max(amount[i - 1], amount[i - 2] + nums[i])
        val amount = IntArray(nums.size)
        amount[0] = nums[0]
        if (nums.size >= 2) {
            amount[1] = maxOf(amount[0], nums[1])
        }

        for (i in 2 until nums.size) {
            amount[i] = maxOf(amount[i - 1], amount[i - 2] + nums[i])
        }

        return amount[nums.size - 1]
    }
}

class Solution198_20220529 {
    fun rob(nums: IntArray): Int {
        var amount = IntArray(nums.size)
        amount[0] = nums[0]
        if (nums.size > 1) {
            amount[1] = maxOf(nums[0], nums[1])
        }
        for (i in 2 until nums.size) {
            amount[i] = maxOf(amount[i - 1], amount[i - 2] + nums[i])
        }
        return amount[nums.size - 1]
    }
}

/// 213. 打家劫舍 II
class Solution213 {
    fun rob(nums: IntArray): Int {
        // dp[i] 代表在第 0~i 这 i+1 栋房屋可以获得的最大金额

        // dp[0] = nums[0]
        // dp[1] = maxOf(nums[0], nums[1])

        val n = nums.size
        if (n == 0) return 0
        if (n <= 3) {
            return maxOf(nums[0], nums.getOrElse(1) { 0 }, nums.getOrElse(2) { 0 })
        }

        // dpNoFirst[i] 代表在第 0~i 这 i+1 栋房屋，并在不偷第一栋房屋的前提下可以获得的最大金额
        var dpNoFirst = IntArray(n)
        dpNoFirst[0] = 0
        dpNoFirst[1] = nums[1]

        var dp = IntArray(n)
        dp[0] = nums[0]
        dp[1] = maxOf(nums[1], nums[0])

        for (i in 2 until n) {
            dpNoFirst[i] = maxOf(nums[i] + dpNoFirst[i - 2], dpNoFirst[i - 1])

            if (i == n - 1) {
                val a = dpNoFirst[i] // dpNoFirst 不偷第一家的前提下的最大值
                val b = dp[i - 1]   // dp 偷第一家的前提下，不偷最后一家时的最大值
                dp[i] = maxOf(a, b)
            } else {
                dp[i] = maxOf(nums[i] + dp[i - 2], dp[i - 1])
            }
        }

        return dp[n - 1]
    }

    fun rob1(nums: IntArray): Int {
        // dp[i][j] 代表小偷从房屋 i 到 j 总共能够获取的最大金额
        /*  dp[i][j] =
            maxOf(
                nums[i] + dp[i + 2][j],
                nums[i + 1] + dp[i + 3][j]
                ...
                dp[i][i + k - 2] + nums[i + k] + dp[i + k + 2][j]
            )
            i <= j

            显然，dp[i][i] = nums[i]
        */
        var dp = Array(nums.size) { IntArray(nums.size) }

        for (i in dp.indices) {
            dp[i][i] = nums[i]
        }

        fun getDp(i: Int, j: Int): Int {
            if (i <= j) {
                return dp[i][j]
            }
            return 0
        }

        for (i in nums.size - 2 downTo 0) {
            for (j in i + 1 until nums.size) {
                var max = 0
                for (k in i until j + 1) {
                    var value = nums[k]
                    // 根据 k 的取值分情况处理
                    if (k == 0) { ////////////////////////// (1). k == 0
                        value += getDp(k + 2, minOf(j, nums.size - 2))
                    } else if (k == nums.size - 1) { /////// (2). k == nums.size - 1
                        value += getDp(maxOf(1, i), k - 2)
                    } else
                    /////////////////////////////// (3). 0 < k < nums.size - 1
                    // 根据 i, j 是否联通，分为 2 种情况
                    {
                        if (i == 0 && j == nums.size - 1) { //// (3.1) 首尾联通时，分为 『取首去尾』 和 『取尾去首』两种情况处理。
                            // 需要考虑首尾联通问题
                            value += maxOf(
                                // 取首去尾
                                getDp(i, k - 2) + getDp(k + 2, minOf(j, nums.size - 2)),
                                // 取尾去首
                                getDp(maxOf(i, 1), k - 2) + getDp(k + 2, j)
                            )
                        } else { //////////////////////////////// (3.2) 首尾不联通时，按普通情况处理
                            value += getDp(i, k - 2) + getDp(k + 2, j)
                        }
                    }
                    max = maxOf(max, value)
                }
                dp[i][j] = max
            }
        }

        return dp[0][nums.size - 1]
    }
}

/// 213. 打家劫舍 II
class Solution213_20220606 {
    fun rob(nums: IntArray): Int {
        fun rob(from: Int, to: Int): Int {
            val n = to - from + 1
            if (n == 1) return nums[from]

            // dp[i] 代表打劫第 from 到 i 家可以得到的最大金额
            // dp[i] = maxOf(dp[i - 2] + nums[i], dp[i - 1])
            // dp[0] = nums[0]
            // dp[1] = maxOf(nums[0], nums[1])
            val dp = IntArray(n)
            dp[0] = nums[from]
            if (from + 1 <= to) {
                dp[1] = maxOf(nums[from], nums[from + 1])
            }
            for (i in 2 until n) {
                dp[i] = maxOf(dp[i - 1], dp[i - 2] + nums[from + i])
            }
            return dp[n - 1]
        }
        if (nums.size == 1) return nums[0]
        return maxOf(rob(0, nums.size - 2), rob(1, nums.size - 1))
    }
}

//class Solution {
//    Map<TreeNode, Integer> f = new HashMap<TreeNode, Integer>();
//    Map<TreeNode, Integer> g = new HashMap<TreeNode, Integer>();
//
//    public int rob(TreeNode root) {
//        dfs(root);
//        return Math.max(f.getOrDefault(root, 0), g.getOrDefault(root, 0));
//    }
//
//    public void dfs(TreeNode node) {
//        if (node == null) {
//            return;
//        }
//        dfs(node.left);
//        dfs(node.right);
//        f.put(node, node.val + g.getOrDefault(node.left, 0) + g.getOrDefault(node.right, 0));
//        g.put(node, Math.max(f.getOrDefault(node.left, 0), g.getOrDefault(node.left, 0)) + Math.max(f.getOrDefault(node.right, 0), g.getOrDefault(node.right, 0)));
//    }
//}

/// 337. 打家劫舍 III
class Solution337 {
    class TreeNode(var `val`: Int) {
        var left: TreeNode? = null
        var right: TreeNode? = null
    }

    //// dfs

    fun dfs(root: TreeNode?): Pair<Int, Int> {
        if (root == null) return Pair<Int, Int>(0, 0)

        val l = dfs(root.left)
        val r = dfs(root.right)
        // first 存选root的情况
        // second 存不选root的情况
        val selectRoot = root.`val` + l.second + r.second
        val notSelectRoot = maxOf(l.first, l.second) + maxOf(r.first, r.second)
        return Pair<Int, Int>(selectRoot, notSelectRoot)
    }

    fun rob(root: TreeNode?): Int {
        val result = dfs(root)
        return maxOf(result.first, result.second)
    }

    /////////////

    /// 记忆化递归 双缓存
    val robRootCache = mutableMapOf<TreeNode?, Int>()
    val noRobRootCache = mutableMapOf<TreeNode?, Int>()

    fun robMax2(root: TreeNode?, canRobRoot: Boolean): Int {
        if (canRobRoot && robRootCache.containsKey(root)) {
            return robRootCache[root]!!
        }
        if (!canRobRoot && noRobRootCache.containsKey(root)) {
            return noRobRootCache[root]!!
        }

        if (root == null) {
            robRootCache[root] = 0
            noRobRootCache[root] = 0
            return 0
        }

        if (canRobRoot) {
            return maxOf(
                root.`val` + robMax2(root.left, false) + robMax2(root.right, false),
                robMax2(root.left, true) + robMax2(root.right, true)
            ).also { robRootCache[root] = it }
        } else {
            return (robMax2(root.left, true) + robMax2(
                root.right,
                true
            )).also { noRobRootCache[root] = it }
        }
    }

    fun rob2(root: TreeNode?): Int {
        return maxOf(robMax2(root, true), robMax2(root, false))
    }

    //////////////

    /// 记忆化递归 单缓存
    fun robMax1(
        root: TreeNode?,
        canRobRoot: Boolean,
        mem: MutableMap<TreeNode?, MutableList<Int?>>
    ): Int {
        fun getCache(): Int? {
            var maxList: MutableList<Int?>
            if (!mem.containsKey(root)) {
                maxList = mutableListOf<Int?>(null, null)
                mem[root] = maxList
            } else {
                maxList = mem[root]!!
            }
            val index = if (canRobRoot) 1 else 0
            return maxList[index]
        }

        fun setCache(maxValue: Int, canRob: Boolean? = null) {
            var maxList: MutableList<Int?>
            if (!mem.containsKey(root)) {
                maxList = mutableListOf<Int?>(null, null)
                mem[root] = maxList
            } else {
                maxList = mem[root]!!
            }
            var canRob = canRob
            if (canRob == null) {
                canRob = canRobRoot
            }
            val index = if (canRob) 1 else 0
            maxList[index] = maxValue
        }


        val cacheValue = getCache()
        if (cacheValue != null) {
            return cacheValue
        }

        if (root == null) {
            setCache(0, true)
            setCache(0, false)
            return 0
        }

        var result = 0
        if (canRobRoot) {
            // 1. rob root & not rob left and right
            val rootMax =
                root.`val` + robMax1(root.left, false, mem) + robMax1(root.right, false, mem)

            // 2. not rob root & can rob left and right
            val subMax = robMax1(root.left, true, mem) + robMax1(root.right, true, mem)
            result = maxOf(rootMax, subMax)
        } else {
            val subMax = robMax1(root.left, true, mem) + robMax1(root.right, true, mem)
            result = subMax
        }

        return result.also { setCache(it) }
    }

    fun rob1(root: TreeNode?): Int {
        var mem = mutableMapOf<TreeNode?, MutableList<Int?>>()
        return robMax1(root, true, mem)
    }
}

/// 18. 四数之和
class Solution18 {
    fun fourSum1(nums: IntArray, target: Int): List<List<Int>> {
        nums.sort()

        val n = nums.size

        var a = 0
        var d = n - 1

        var result = mutableSetOf<List<Int>>()

        while (a <= d - 3) {
            var b = a + 1
            var c = d - 1

            fun process() {
                while (b < c) {
                    if (nums[a] + nums[c - 1] + nums[c] + nums[d] < target ||
                        nums[a] + nums[b] + nums[b + 1] + nums[d] > target
                    ) {
                        return
                    }

                    val sum = nums[a] + nums[b] + nums[c] + nums[d]
                    if (sum == target) {
                        result.add(listOf(nums[a], nums[b], nums[c], nums[d]))
                        while (b < c && nums[b + 1] == nums[b]) {
                            b++
                        }
                        b++
                        while (b < c && nums[c - 1] == nums[c]) {
                            c--
                        }
                        c++
                    } else if (sum < target) {
                        b++
                    } else { // sum > target
                        c--
                    }
                }
            }

            process()

            var tmp = d

            while (a <= d - 3) {
                b = a + 1
                d -= 1
                c = d - 1
                if (a <= d - 3) {
                    process()
                }
            }

            d = tmp
            tmp = a

            while (a <= d - 3) {
                a += 1
                b = a + 1
                c = d - 1
                if (a <= d - 3) {
                    process()
                }
            }

            a = tmp + 1
            d -= 1
        }

        return result.toList()
    }

    fun fourSumO(nums: IntArray, target: Int): List<List<Int>> {
        //4数之和，前面3数之和可以使用 排序 + 双指针 来实现，这个4数之和，感觉可以
        //2层遍历 加双指针

        //特判
        if (nums.size < 4) {
            return ArrayList()
        }

        //定义2个指针
        var leftIndex: Int
        var rightIndex: Int

        //定义返回集合，使用set来去重
        val ans = ArrayList<ArrayList<Int>>()

        //先排序
        Arrays.sort(nums)

        //第一层遍历
        for (i in 0..nums.size - 4) {
            //优化特判 当i从1开始，假如nums[1]和nums[0]相同，则i==1不用再处理
            if (i > 0 && nums[i] == nums[i - 1]) continue

            //优化特判 当前几个最小值已经大于target，则无需再遍历
            if (nums[i].toLong() + nums[i + 1].toLong() + nums[i + 2].toLong() + nums[i + 3].toLong() > target) {
                break
            }
            //优化特判 当当前值和最大的几个值小于target，则说明i要往后面移动了
            if (nums[i].toLong() + nums[nums.size - 1].toLong() + nums[nums.size - 2].toLong() + nums[nums.size - 3].toLong() < target) {
                continue
            }
            //第二层遍历
            for (j in i + 1..nums.size - 3) {
                //优化特例 当当前j的值和上一个j值一样，则不处理
                if (j > i + 1 && nums[j] == nums[j - 1]) {
                    continue
                }
                //优化特判 当前几个最小值都已经大于target，则无需再遍历
                if (nums[i].toLong() + nums[j].toLong() + nums[j + 1].toLong() + nums[j + 2].toLong() > target) {
                    break
                }
                //优化特判 当当前值和最大的几个值小于target，说明j需要右移动了
                if (nums[i].toLong() + nums[j].toLong() + nums[nums.size - 1].toLong() + nums[nums.size - 2].toLong() < target) {
                    continue
                }
                //再使用双指针
                leftIndex = j + 1
                rightIndex = nums.size - 1
                //循环判断
                while (leftIndex < rightIndex) {
                    val curSum = nums[i] + nums[j] + nums[leftIndex] + nums[rightIndex]
                    when {
                        curSum == target -> {
                            //符合条件，返回值
                            val tempList =
                                arrayListOf(nums[i], nums[j], nums[leftIndex], nums[rightIndex])
                            ans.add(tempList)
                            //优化特判 当左指针右移时，可能移动完和前面是一样的
                            leftIndex++
                            while (leftIndex < rightIndex && nums[leftIndex] == nums[leftIndex - 1]) {
                                leftIndex++
                            }
                            //优化特判 当右指针左移时，可能移动完和前面是一样的
                            rightIndex--
                            while (leftIndex < rightIndex && nums[rightIndex] == nums[rightIndex + 1]) {
                                rightIndex--
                            }
                        }
                        curSum > target -> {
                            //4数之和大于target
                            rightIndex--
                        }
                        curSum < target -> {
                            //4数之和小于target
                            leftIndex++
                        }
                    }
                }
            }
        }
        return ans
    }

    fun fourSum(nums: IntArray, target: Int): List<List<Int>> {
        if (nums.size < 4) return listOf()

        val result = mutableListOf<List<Int>>()
        val n = nums.size

        nums.sort()
        if (nums[n - 1].toLong() + nums[n - 2] + nums[n - 3] + nums[n - 4] < target ||
            nums[0].toLong() + nums[1] + nums[2] + nums[3] > target
        ) {
            return result
        }

        for (a in 0 until n - 3) {
            if (a > 0 && nums[a] == nums[a - 1]) continue

            if (nums[a].toLong() + nums[a + 1] + nums[a + 2] + nums[a + 3] > target) break
            if (nums[a].toLong() + nums[n - 1] + nums[n - 2] + nums[n - 3] < target) continue

            for (b in a + 1 until n - 2) {
                if (b > a + 1 && nums[b] == nums[b - 1]) continue

                if (nums[a].toLong() + nums[b] + nums[b + 1] + nums[b + 2] > target) break
                if (nums[a].toLong() + nums[b] + nums[n - 1] + nums[n - 2] < target) continue

                var d = n - 1
                for (c in b + 1 until n - 1) {
                    if (c > b + 1 && nums[c] == nums[c - 1]) continue

                    val sum = nums[a].toLong() + nums[b] + nums[c]
                    if (sum + nums[c + 1].toLong() > target) break
                    if (sum + nums[d].toLong() < target) continue

                    while (c < d && sum + nums[d] > target) d--
                    if (c == d) break

                    if (sum + nums[d] == target.toLong()) {
                        result.add(listOf(nums[a], nums[b], nums[c], nums[d]))
                    }
                }
            }
        }

        return result
    }

}

/// 32. 最长有效括号
class Solution32 {
    fun longestValidParenthesesTwoRuns(s: String): Int {
        var l = 0
        var r = 0
        var max = 0

        for (c in s) {
            if (c == '(') {
                l++
            } else {
                r++
            }

            if (r > l) {
                l = 0
                r = 0
            }

            if (l == r && l * 2 > max) {
                max = l * 2
            }
        }

        l = 0
        r = 0

        for (c in s.reversed()) {
            if (c == '(') {
                l++
            } else {
                r++
            }

            if (r < l) {
                l = 0
                r = 0
            }

            if (l == r && l * 2 > max) {
                max = l * 2
            }
        }

        return max
    }

    fun longestValidParenthesesDP(s: String): Int {
        var n = s.length
        if (n < 2) return 0

        // dp[i] 代表以 s[i] 结尾的最长有效括号子串的长度，所以所有 '(' 对应的 dp[i] = 0，因为以 '(' 结尾的子串不可能是有效括号子串。
        var dp = IntArray(n)
        // 默认 dp[0] = 0, dp[1] 仅在 s[0] == '(' && s[1] == ')' 时为 2，否则为 0
        dp[1] = if (s[0] == '(' && s[1] == ')') 2 else 0

        var max = dp[1]

        for (i in 2 until n) {
            if (s[i] == ')') {
                if (s[i - 1] == '(') {
                    dp[i] = dp[i - 2] + 2
                } else if (s[i - 1] == ')') {
                    var c = i - dp[i - 1] - 1
                    if (c >= 0 && s[c] == '(') {
                        dp[i] = 2 + dp[i - 1]
                        if (c - 1 >= 0) {
                            dp[i] += dp[c - 1]
                        }
                    }
                }
                if (max < dp[i]) {
                    max = dp[i]
                }
            }
        }

        return max
    }
}

/// 448. 找到所有数组中消失的数字
class Solution448 {
    fun findDisappearedNumbers1(nums: IntArray): List<Int> {
        var seen = mutableMapOf<Int, Boolean>()
        for (i in nums) {
            seen[i] = true
        }
        var result = mutableListOf<Int>()
        for (i in 1 until nums.size + 1) {
            if (!seen.containsKey(i)) {
                result.add(i)
            }
        }
        return result
    }

    fun findDisappearedNumbers2(nums: IntArray): List<Int> {
        var result = mutableListOf<Int>()

        nums.sort()

        var target = 1
        var i = 0
        while (i < nums.size && target <= nums.size) {
            if (target == nums[i]) {
                i++
                while (i < nums.size && nums[i - 1] == nums[i]) {
                    i++
                }
                target++
            } else if (nums[i] > target) {
                result.add(target)
                target++
            } else { // nums[i] < target
                i++
            }
        }

        while (target <= nums.size) {
            result.add(target)
            target++
        }

        return result
    }

    /// 将所有可以恢复的数字恢复到原位，然后再遍历一遍找到值和下标不对应的元素。
    fun findDisappearedNumbers(nums: IntArray): List<Int> {
        var i = 0
        while (i < nums.size) {
            val l = nums[i]
            if (l != i + 1) {
                if (nums[i] == nums[l - 1]) {
                    i++
                } else {
                    nums[l - 1] = nums[i].also { nums[i] = nums[l - 1] }
                }
            } else {
                i++
            }
        }

        var result = mutableListOf<Int>()
        for (i in nums.indices) {
            if (nums[i] != i + 1) {
                result.add(i + 1)
            }
        }

        return result
    }

    /// 原地哈希标记法
    fun findDisappearedNumbersHASH(nums: IntArray): List<Int> {
        val n = nums.size
        for (num in nums) {
            nums[(num - 1) % n] += n
        }
        var result = mutableListOf<Int>()
        for (i in nums.indices) {
            if (nums[i] <= n) {
                result.add(i + 1)
            }
        }
        return result
    }
}

/// 953. 验证外星语词典
class Solution953 {
    fun isAlienSorted(words: Array<String>, order: String): Boolean {
        /// 记录每个字符应该处于的下标，下标越小其字典序越靠前
        var orderInfo = ShortArray(26)
        for (i in order.indices) {
            orderInfo[order[i] - 'a'] = i.toShort()
        }

        val MIN: Short = -1
        var valid = true
        for (i in 1 until words.size) {
            // 检查 w1 和 w2 是否是order字典序
            val w1 = words[i - 1]
            val w2 = words[i]
            var i1 = 0
            var i2 = 0
            while (i1 < w1.length || i2 < w2.length) {
                var v1 = if (i1 < w1.length) orderInfo[w1[i1] - 'a'] else MIN
                var v2 = if (i2 < w2.length) orderInfo[w2[i2] - 'a'] else MIN

                if (v2 > v1) {
                    break
                } else if (v1 > v2) {
                    valid = false // 发现一处非 orderd字典序
                    break
                }

                i1++
                i2++
            }
        }

        return valid
    }
}

/// 567. 字符串的排列
class Solution567 {

    fun checkInclusion(s1: String, s2: String): Boolean {
        val feq = IntArray(26)
        var cnt = 0
        val n1 = s1.length
        val n2 = s2.length
        for (c in s1.toCharArray()) {
            if (feq[c - 'a'] == 0) {
                cnt++
            }
            feq[c - 'a']++
        }
        val chars = s2.toCharArray()
        for (i in 0 until n2) {
            if (--feq[chars[i] - 'a'] == 0) {
                cnt--
            }
            if (i >= n1 && feq[chars[i - n1] - 'a']++ == 0) {
                cnt++
            }
            if (cnt == 0) {
                return true
            }
        }
        return false
    }


    fun checkInclusion1(s1: String, s2: String): Boolean {
        var s1Indices = IntArray(26)
        for (c in s1) {
            s1Indices[c - 'a']++
        }

        var s2Indices = IntArray(26)

        fun indicesEqual(): Boolean {
            var equal = true
            for (i in 0 until s1Indices.size) {
                if (s1Indices[i] != s2Indices[i]) {
                    equal = false
                    break
                }
            }
            return equal
        }

        var i = 0
        while (i < s2.length) {
            val c = s2[i] - 'a'
            if (s1Indices[c] > 0) {
                s2Indices[c]++
            }
            if (i >= s1.length) {
                val dump = s2[i - s1.length] - 'a'
                if (s2Indices[dump] > 0) {
                    s2Indices[dump]--
                }
            }
            if (i >= s1.length - 1) {

                if (indicesEqual()) {
                    return true
                }
            }
            i++
        }

        return false
    }
}

/// 20. 有效的括号
/*
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：
 - 左括号必须用相同类型的右括号闭合。
 - 左括号必须以正确的顺序闭合。

*/
class Solution20 {
    fun isValid1(s: String): Boolean {
        fun isLeft(c: Char) = (c == '(' || c == '{' || c == '[')
        fun isRight(c: Char) = (c == ')' || c == '}' || c == ']')
        fun isPair(left: Char, right: Char) =
            (left == '(' && right == ')' || left == '{' && right == '}' || left == '[' && right == ']')

        var stack = mutableListOf<Char>('a')
        for (c in s) {
            if (isLeft(c)) {
                stack.add(c)
            } else if (isRight(c)) {
                if (isPair(stack.last(), c)) {
                    stack.removeAt(stack.size - 1)
                } else {
                    return false
                }
            }
        }

        return stack.size == 1
    }

    fun isValid(s: String): Boolean {
        if (s.length % 2 == 1) return false

        var stack = LinkedList<Char>()
        stack.add('_')
        for (c in s) {
            when (c) {
                '(' -> stack.addLast(')')
                '{' -> stack.addLast('}')
                '[' -> stack.addLast(']')
                else -> {
                    if (stack.removeLast() != c) return false
                }
            }
        }

        return stack.size == 1
    }
}

class Solution22_20220607 {
    fun isValid(s: String): Boolean {
        if (s.length and 1 == 1) return false

        val stack = Stack<Char>()
        for (c in s) {
            when (c) {
                '(' -> stack.push(')')
                '{' -> stack.push('}')
                '[' -> stack.push(']')
                else -> {
                    if (stack.isEmpty() || stack.pop() != c) {
                        return false
                    }
                }
            }
        }
        return stack.isEmpty()
    }
}

/// 22. 括号生成
class Solution22 {
    fun generateParenthesis(n: Int): List<String> {
        val res = generate(n * 2)
        val res1 = res.filter { isValid(it) }
        return res1
    }

    fun isValid(s: String): Boolean {
        var count = 0
        for (c in s) {
            when (c) {
                '(' -> count++
                else -> {
                    count--
                    if (count < 0) return false
                }
            }
        }
        return count == 0
    }

    fun generate(n: Int): List<String> {
        if (n == 1) return listOf("(", ")")

        var results = mutableListOf<String>()

        var subResults = generate(n - 1)
        subResults.forEach {
            var r = ")" + it
            results.add(r)
            r = "(" + it
            results.add(r)
        }

        return results
    }
}

class Solution22_1 {
    fun generateParenthesis(n: Int): List<String> {
        val results = mutableListOf<String>()
        generate(CharArray(n * 2), 0, results)
        return results
    }

    fun isValid(s: CharArray): Boolean {
        var count = 0
        for (c in s) {
            when (c) {
                '(' -> count++
                else -> {
                    count--
                    if (count < 0) return false
                }
            }
        }
        return count == 0
    }

    fun generate(chars: CharArray, pos: Int, result: MutableList<String>) {
        if (chars.size == pos) {
            if (isValid(chars)) {
                result.add(chars.joinToString(""))
            }
        } else {
            chars[pos] = '('
            generate(chars, pos + 1, result)
            chars[pos] = ')'
            generate(chars, pos + 1, result)
        }
    }
}

class Solution22_2 {
    fun generateParenthesis(n: Int): List<String> {
        val results = mutableListOf<String>()
        generate(CharArray(n * 2), 0, 0, 0, results)
        return results
    }

    fun generate(chars: CharArray, pos: Int, left: Int, right: Int, result: MutableList<String>) {
        if (chars.size == pos) {
            result.add(chars.joinToString(""))
        } else {
            var max = chars.size / 2
            if (left < max) {
                chars[pos] = '('
                generate(chars, pos + 1, left + 1, right, result)
            }

            if (left > right) {
                chars[pos] = ')'
                generate(chars, pos + 1, left, right + 1, result)
            }
        }
    }
}

class Solution22_3 {
    fun generateParenthesis(n: Int): List<String> {
        val res = ArrayList<String>()
        dfs(res, n, n, "")
        return res
    }

    private fun dfs(
        res: ArrayList<String>,
        left: Int, right: Int, curStr: String
    ) {
        if (left == 0 && right == 0) {
            res.add(curStr)
            return
        }
        if (left > 0) {
            dfs(res, left - 1, right, curStr + "(")
        }
        if (right > left) {
            dfs(res, left, right - 1, curStr + ")")
        }
    }
}

class Solution22_20220605 {
    fun generateParenthesis(n: Int): List<String> {
        val results = mutableListOf<String>()
        fun dfs(left: Int, right: Int, path: String) {
            if (left == 0 && right == 0) {
                results.add(path)
            }

            if (left > 0) {
                dfs(left - 1, right, path + "(")
            }
            if (left < right) {
                dfs(left, right - 1, path + ")")
            }
        }
        dfs(n, n, "")
        return results
    }
}

/// 136. 只出现一次的数字
class Solution136 {
    fun singleNumber(nums: IntArray): Int {
        var result = 0
        for (num in nums) {
            result = result xor num
        }
        return result
    }

    fun singleNumber1(nums: IntArray): Int {
        nums.sort()
        var result = 0
        for (i in 0 until nums.size) {
            if (i % 2 == 0) {
                result += nums[i]
            } else {
                result -= nums[i]
            }
        }
        return result
    }
}

/// 146. LRU 缓存
/*
请你设计并实现一个满足 LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：
- LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
- int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
- void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组key-value 。
  如果插入操作导致关键字数量超过capacity ，则应该 逐出 最久未使用的关键字。

函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。
* */
class LRUCache(val capacity: Int) {
    private class Node(
        var value: Int,
        val key: Int,
        var next: Node? = null,
        var prev: Node? = null
    )

    private var keyValues = mutableMapOf<Int, Node>()

    private var preHead = Node(-1, -1)
    private var postTail = Node(-2, -2)

    init {
        preHead.next = postTail
        postTail.prev = preHead
    }

    private fun insertAtFirst(node: Node) {
        val next = preHead.next!!

        preHead.next = node
        node.prev = preHead

        node.next = next
        next.prev = node
    }

    private fun removeNode(node: Node) {
        if (node.next == null || node.prev == null) return

        val prev = node.prev!!
        val next = node.next!!

        node.next = null
        node.prev = null

        prev.next = next
        next.prev = prev
    }

    fun get(key: Int): Int {
        val node = keyValues.get(key)
        if (node != null) {
            if (node != preHead.next) {
                removeNode(node)
                insertAtFirst(node)
            }
            return node.value
        }
        return -1
    }

    fun put(key: Int, value: Int) {
        var node: Node
        if (keyValues.containsKey(key)) {
            node = keyValues[key]!!
            node.value = value

            removeNode(node)
            insertAtFirst(node)
        } else {
            node = Node(value, key)
            keyValues[key] = node
            insertAtFirst(node)

            if (keyValues.size > capacity) {
                // 链表中删除最后一个元素 postTail.prev
                val deleted = postTail.prev!!
                removeNode(deleted)

                // 哈希表中删除对应的 key 值
                keyValues.remove(deleted.key)
            }
        }
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * var obj = LRUCache(capacity)
 * var param_1 = obj.get(key)
 * obj.put(key,value)
 */

/// 733. 图像渲染
class Solution733 {
    fun floodFill(image: Array<IntArray>, sr: Int, sc: Int, newColor: Int): Array<IntArray> {
        val color = image[sr][sc]
        if (color == newColor) return image


        val rMax = image.size - 1
        val cMax = image[0].size - 1

        var fillPoints = LinkedList<Pair<Int, Int>>()
        fillPoints.offer(Pair(sr, sc))
        image[sr][sc] = newColor

        fun fillPointIfNeeded(r: Int, c: Int) {
            if (color == image[r][c]) {
                fillPoints.add(Pair(r, c))
                image[r][c] = newColor
            }
        }

        while (fillPoints.isNotEmpty()) {
            val p = fillPoints.poll()
            val (r, c) = p
            // 检查s上下左右的点，如果颜色相同的点不在 filledPoints 中，则将其加入 fillPoints
            // 上
            if (r - 1 >= 0) {
                fillPointIfNeeded(r - 1, c)
            }
            // 下
            if (r + 1 <= rMax) {
                fillPointIfNeeded(r + 1, c)
            }
            // 左
            if (c - 1 >= 0) {
                fillPointIfNeeded(r, c - 1)
            }
            // 右
            if (c + 1 <= cMax) {
                fillPointIfNeeded(r, c + 1)
            }
        }

        return image
    }
}

/// 695. 岛屿的最大面积
class Solution695 {
    fun maxAreaOfIsland(grid: Array<IntArray>): Int {
        val rowCount = grid.size
        val columnCount = grid[0].size

        var searchedPoints = mutableSetOf<Int>()
        var maxArea = 0

        fun getKey(row: Int, column: Int) = row * columnCount + column
        fun getRowColumn(key: Int) = Pair(key / columnCount, key % columnCount)

        fun searchMax(row: Int, column: Int): Int {
            val pointKey = getKey(row, column)
            var max = 0

            if (grid[row][column] == 1 && !searchedPoints.contains(pointKey)) {
                val willSearchPoints = LinkedList<Int>()
                willSearchPoints.offer(pointKey)

                max++
                searchedPoints.add(pointKey)
                while (willSearchPoints.isNotEmpty()) {
                    val key = willSearchPoints.poll()!!
                    val (r, c) = getRowColumn(key)
                    // 向上下左右扩散搜索为 1 的区域
                    // up
                    fun subSearch(key: Int) {
                        if (!searchedPoints.contains(key)) {
                            willSearchPoints.offer(key)
                            searchedPoints.add(key)
                            max++
                        }
                    }

                    if (r - 1 >= 0 && grid[r - 1][c] == 1) {
                        subSearch(getKey(r - 1, c))
                    }
                    // down
                    if (r + 1 < rowCount && grid[r + 1][c] == 1) {
                        subSearch(getKey(r + 1, c))
                    }
                    // left
                    if (c - 1 >= 0 && grid[r][c - 1] == 1) {
                        subSearch(getKey(r, c - 1))
                    }
                    // right
                    if (c + 1 < columnCount && grid[r][c + 1] == 1) {
                        subSearch(getKey(r, c + 1))
                    }
                }
            }

            return max
        }

        // 一行一行遍历
        for (i in 0 until rowCount) {
            for (j in 0 until columnCount) {
                maxArea = maxOf(maxArea, searchMax(i, j))
            }
        }

        return maxArea
    }

    ///// 递归

    fun maxAreaOfIslandRC(grid: Array<IntArray>): Int {
        var maxSize = 0
        for (i in grid.indices) {
            for (j in grid[i].indices) {
                if (grid[i][j] == 1) {
                    var maxLen = grid.findMax(i, j)
                    maxSize = Math.max(maxLen, maxSize)
                }
            }
        }
        return maxSize
    }

    fun Array<IntArray>.findMax(i: Int, j: Int): Int {
        if (i < 0 || i >= this.size || j < 0 || j >= this[i].size || this[i][j] != 1) {
            return 0
        }
        this[i][j] = 0
        return 1 + findMax(i + 1, j) + findMax(i - 1, j) + findMax(i, j - 1) + findMax(i, j + 1)
    }

    fun maxAreaOfIslandGood(grid: Array<IntArray>): Int {
        val rowMax = grid.size - 1
        val colMax = grid[0].size - 1
        var max = 0

        fun countAndWipeLand(i: Int, j: Int): Int {
            if (i < 0 || i > rowMax || j < 0 || j > colMax || grid[i][j] == 0) return 0
            grid[i][j] = 0
            return 1 + countAndWipeLand(i + 1, j) + countAndWipeLand(i - 1, j) + countAndWipeLand(
                i,
                j + 1
            ) + countAndWipeLand(i, j - 1)
        }

        for (i in 0 until rowMax + 1) {
            for (j in 0 until colMax + 1) {
                max = maxOf(max, countAndWipeLand(i, j))
            }
        }

        return max
    }
}

class Solution695_20220531 {
    fun maxAreaOfIsland(grid: Array<IntArray>): Int {
        val rowCount = grid.size
        val colCount = grid[0].size

        fun wipeAndCount(row: Int, col: Int): Int {
            if (row < 0 || row >= rowCount || col < 0 || col >= colCount || grid[row][col] == 0) return 0

            grid[row][col] = 0
            return 1 + wipeAndCount(row + 1, col) + wipeAndCount(row - 1, col) + wipeAndCount(
                row,
                col + 1
            ) + wipeAndCount(row, col - 1)
        }

        var maxCount = 0
        for (r in 0 until rowCount) {
            for (c in 0 until colCount) {
                maxCount = maxOf(maxCount, wipeAndCount(r, c))
            }
        }

        return maxCount
    }
}

/// 215. 数组中的第K个最大元素
class Solution215 {
    /*
    - pivot 是一个分界，这边直接取了最后一位(下标r)
    - j 只是用来遍历取完分界后的数组，目标就是直接把小于 pivot 的数往前挪，自然最好的方式是直接依次挪到第1，2，3，4...位
    - i 最终代表 pivot 所处的下标，所以最终需要 Swap A[i] and A[r]
    * */
    fun partitionAscend(nums: IntArray, start: Int, end: Int): Int {
        fun swap(i: Int, j: Int) {
            if (i == j) return
            nums[i] = nums[j].also { nums[j] = nums[i] }
        }

        val pivot = nums[end]
        var i = start
        for (j in start until end) {
            if (nums[j] < pivot) { // pivot = nums[end]
                swap(i, j) // 将比 pivot 小的数放到位置 i，i自然需要往后移动一位
                i++
            }
            // 如果本次循环 nums[j] 比 pivot 大，则 i 保持不变，而 j 每次循环结束后都往后移动一位
            // 所以下标指针 j 总是比下标指针 i 移动得快。
        }

        // 遍历完 start ~> end - 1 所有数字，此时可以确认 i 下标的数是不小于 pivot 的，所以将下标位置 i 的数和 pivot 交换
        swap(i, end)
        return i
    }

    /// 给 nums 范围 [start, end] 进行升序分块，返回分区点下标
    fun partitionAssend_20220603(nums: IntArray, start: Int = 0, end: Int = nums.size - 1): Int {
        val pivot = nums[end]
        var i = 0
        for (j in start until end) {
            if (nums[j] < pivot) {
                if (i != j) nums[i] = nums[j].also { nums[j] = nums[i] }
                i++
            }
        }
        // 此时下标i对应的数字应该是大于或等于 pivot 的，所以做一次交换
        if (i != end) nums[i] = nums[end].also { nums[end] = nums[i] }
        return i
    }

    val rand = Random()
    fun partitionDescend(nums: IntArray, start: Int, end: Int): Int {
        fun swap(i: Int, j: Int) {
            if (i == j) return
            nums[i] = nums[j].also { nums[j] = nums[i] }
        }

        // 随机取 pivot，避免极端情况
        val pivotIndex = rand.nextInt(end - start + 1) + start
        swap(pivotIndex, end)

        val pivot = nums[end]
        var i = start
        for (j in start until end) {
            if (nums[j] > pivot) {
                swap(i, j)
                i++
            }
        }
        swap(i, end)
        return i
    }

    fun findKthLargest(nums: IntArray, k: Int): Int {
        // 使用快排分区的思想，第一次选一个数 x，把剩下的数中比 x 大的放在 x 左侧，比 x 小的放在 x 右侧，
        // 最终可以知道 x 的所处位置 ix，x 即为数组中第 ix + 1 大的元素，因为共有 ix 个元素比 x 大。
        // 如果 k == ix + 1，则 x 即为所找的数 => ix == k - 1
        var k = k
        var start = 0
        var end = nums.size - 1
        while (start <= end) {
            val pivotIndex = partitionDescend(nums, start, end)
            if (pivotIndex == k - 1) {
                return nums[pivotIndex]
            } else if (pivotIndex < k - 1) { // k - 1 是目标
                start = pivotIndex + 1
                k -= (pivotIndex - start + 1) // 排除掉前面 pivotIndex - start + 1 个更大的数
            } else { // k - 1 < pivotIndex
                end = pivotIndex - 1 // 此时排除掉得是比目标数更小的数，所以 k 值不需要变化
            }
        }
        return -1
    }
}

/// 240. 搜索二维矩阵 II
class Solution240 {
    fun searchMatrix(matrix: Array<IntArray>, target: Int): Boolean {
        // top, bottom, left, right 分界数据列表
        var tblrList = mutableListOf(intArrayOf(0, matrix.size - 1, 0, matrix[0].size - 1))
        while (tblrList.isNotEmpty()) {
            val tblr = tblrList.removeAt(0)!!
            var top = tblr[0]
            var bottom = tblr[1]
            var left = tblr[2]
            var right = tblr[3]
            while (top <= bottom && left <= right) {
                val mid = (top + bottom) ushr 1
                val nums = matrix[mid]

                // 找到第一个小于等于 target 的数的下标
                val indexLE = BinarySearch.indexOfLessThanOrEqual(nums, target, left, right)
                if (indexLE == -1) {
                    // 没有找到比 target 小的数，则都比 target 大
                    bottom = mid - 1
                } else if (nums[indexLE] == target) {
                    // 如果就是目标数，则直接返回
                    return true
                } else if (indexLE == right) {
                    // 都比 target 小
                    top = mid + 1
                } else {
                    // index 在 0 ~ colMax-1 范围
                    // 找到第一个大于等于 target 的数的下标
                    val indexGE =
                        BinarySearch.indexOfGreaterThanOrEqual(nums, target, left, right)
                    tblrList.add(intArrayOf(mid + 1, bottom, left, indexGE - 1))
                    tblrList.add(intArrayOf(top, mid - 1, indexLE + 1, right))
                    break
                }
            }
        }

        return false
    }

    fun searchMatrixWoohoo(matrix: Array<IntArray>, target: Int): Boolean {
        val rowMax = matrix.size - 1
        val colMax = matrix[0].size - 1

        // 左下角
        var row = rowMax // 垂直向下增长
        var col = 0 // 水平向右增长

        while (row >= 0 && col <= colMax) {
            val v = matrix[row][col]
            if (v == target) {
                return true
            }

            if (v < target) {
                col++
            } else { // matrix[row][col] > target
                row--
            }
        }

        return false
    }
}

/// 668. 乘法表中第k小的数
class Solution668 {
    fun findKthNumber1(m: Int, n: Int, k: Int): Int {
        var left = 1
        var right = m * n
        while (left < right) {
            var x = left + (right - left) / 2
            var count = x / n * n;
            for (i in x / n + 1 until m + 1) {
                count += x / i
            }
            if (count >= k) {
                right = x
            } else {
                left = x + 1
            }
        }
        return left
    }

    fun findKthNumber(m: Int, n: Int, target: Int): Int {
        fun count(x: Int): Int {
            // 从第 1 行开始，第 i 行的数字：1*i, 2*i, 3*i, ..., n*i, 所以
            // (1) 0 <= x / i <= n, 则第 i 行有 x / i 个数小于等于 x
            // (2) x / i > n, 则第 i 行有 n 个数小于等于 x
            // 总结：count = min(x/i, n)
            var sum = x / n * n
            for (i in x / n + 1 until m + 1) {
                sum += minOf(x / i, n)
            }
            return sum
        }

        var left = 1
        var right = m * n
        while (left < right) {
            val mid = (left + right) ushr 1 // left + (right - left) / 2
            // 要找符合 count(mid) == target 的最小 mid, 所以 target <= count(mid) 时，右边界不收缩
            if (target <= count(mid)) {
                right = mid
            } else { // count(mid) < target, 因为小于，所以目标必然不是 mid，所以左边界 left = mid + 1
                left = mid + 1
            }
        }

        return left
    }
}

/// 17. 电话号码的字母组合
class Solution17 {
    fun letterCombinations(digits: String): List<String> {
        if (digits.length == 0) return listOf()

        var results = mutableListOf<String>()
        var digitMap = mapOf(
            Pair('2', listOf('a', 'b', 'c')),
            Pair('3', listOf('d', 'e', 'f')),
            Pair('4', listOf('g', 'h', 'i')),
            Pair('5', listOf('j', 'k', 'l')),
            Pair('6', listOf('m', 'n', 'o')),
            Pair('7', listOf('p', 'q', 'r', 's')),
            Pair('8', listOf('t', 'u', 'v')),
            Pair('9', listOf('w', 'x', 'y', 'z'))
        )

        fun combine(content: String, i: Int) {
            if (i == digits.length) {
                results.add(content)
                return
            }

            val d = digits[i]
            for (c in digitMap[d]!!) {
                combine(content + c, i + 1)
            }
        }

        combine("", 0)
        return results
    }
}

/// 31. 下一个排列
class Solution31 {
    fun nextPermutation(nums: IntArray): Unit {
        fun swap(l: Int, r: Int) {
            nums[l] = nums[r].also { nums[r] = nums[l] }
        }

        fun flip(from: Int, to: Int) {
            var l = from
            var r = to
            while (l < r) {
                swap(l, r)
                l++
                r--
            }
        }

        /// 从 from ~> to 降序排列，找到从右往左数第一个大于 target 的数的下标
        fun searchGreeterThen(target: Int, from: Int, to: Int): Int {
            var l = from
            var r = to
            while (l <= r) {
                val mid = (l + r) ushr 1
                if (nums[mid] > target && (mid == to || nums[mid + 1] <= target)) {
                    return mid
                } else if (nums[mid] > target) {
                    l = mid + 1
                } else { // target >= nums[mid]
                    r = mid - 1
                }
            }
            return -1
        }

        // 是否降序
        // 最后两个数升序
        // 最后两个数降序

        if (nums.size == 1) return

        var i = nums.size - 1
        while (i > 0) {
            if (nums[i - 1] < nums[i]) {
                // 找到从右往左第一个降序
                val index = searchGreeterThen(nums[i - 1], i, nums.size - 1)
                swap(i - 1, index)
                flip(i, nums.size - 1)
                return
            }
            i--
        }

        // 从左往右，整体降序，则需要翻转整个数组
        flip(0, nums.size - 1)
    }
}

/// 200. 岛屿数量
class Solution200 {
    fun numIslands(grid: Array<CharArray>): Int {
        val rowMax = grid.size - 1
        val colMax = grid[0].size - 1

        fun wipe(row: Int, col: Int): Int {
            if (row < 0 || row > rowMax || col < 0 || col > colMax || grid[row][col] == '0') {
                return 0
            }
            grid[row][col] = '0'
            wipe(row + 1, col)
            wipe(row - 1, col)
            wipe(row, col + 1)
            wipe(row, col - 1)
            return 1
        }

        var count = 0
        for (row in 0 until rowMax + 1) {
            for (col in 0 until colMax + 1) {
                count += wipe(row, col)
            }
        }

        return count
    }
}

class Solution200_20220531 {
    fun numIslands(grid: Array<CharArray>): Int {
        val rowCount = grid.size
        val colCount = grid[0].size

        val dirs = listOf(Pair(1, 0), Pair(-1, 0), Pair(0, 1), Pair(0, -1))
        fun wipe(row: Int, col: Int): Int {
            if (row in 0 until rowCount && col in 0 until colCount && grid[row][col] == '1') {
                grid[row][col] = '0'
                dirs.forEach { (dr, dc) ->
                    wipe(row + dr, col + dc)
                }
                return 1
            }
            return 0
        }

        var count = 0
        for (r in 0 until rowCount) {
            for (c in 0 until colCount) {
                count += wipe(r, c)
            }
        }

        return count
    }
}

/**
 * Example:
 * var ti = TreeNode(5)
 * var v = ti.`val`
 * Definition for a binary tree node.
 *
 */
class TreeNode(var `val`: Int) {
    var left: TreeNode? = null
    var right: TreeNode? = null
}

/// 617. 合并二叉树
class Solution617 {
    fun mergeTrees1(root1: TreeNode?, root2: TreeNode?): TreeNode? {
        if (root1 == null) return root2
        if (root2 == null) return root1

        var node1 = root1
        var node2 = root2
        var root = TreeNode(node1.`val` + node2.`val`)
        root.left = mergeTrees(node1.left, node2.left)
        root.right = mergeTrees(node1.right, node2.right)
        return root
    }

    fun mergeTrees(root1: TreeNode?, root2: TreeNode?): TreeNode? {
        if (root1 == null) return root2
        if (root2 == null) return root1

        var node1 = root1
        var node2 = root2
        node1.`val` += node2.`val`
        node1.left = mergeTrees(node1.left, node2.left)
        node1.right = mergeTrees(node1.right, node2.right)
        return node1
    }
}

/// 116. 填充每个节点的下一个右侧节点指针
class Solution116 {
    class Node(var `val`: Int) {
        var left: Node? = null
        var right: Node? = null
        var next: Node? = null
    }

    fun connect(root: Node?): Node? {
        if (root == null) return root

        var tmpNodes = mutableListOf<Node>()
        var nodes = LinkedList(listOf(root))

        while (nodes.isNotEmpty()) {
            val node = nodes.pollFirst()
            tmpNodes.add(node)

            if (node.left == null) continue

            nodes.add(node.left!!)
            nodes.add(node.right!!)
        }

        var power = 1
        var target = 0
        for (i in 0 until tmpNodes.size - 1) {
            if (i == target) {
                power *= 2
                target += power
            } else {
                tmpNodes[i].next = tmpNodes[i + 1]
            }
        }

        return root
    }

    fun connectRecurve(root: Node?): Node? {

        fun connectNext(node: Node?) {
            if (node?.left == null || node?.right == null) {
                return
            }

            node.left!!.next = node!!.right
            node.right!!.next = node?.next?.left

            connectNext(node.left)
            connectNext(node.right)
        }

        connectNext(root)
        return root
    }

    fun connect3(root: Node?): Node? {

        fun connectNext(node: Node?) {
            if (node == null) return

            if (node.left != null) {
                node.left!!.next = node!!.right
                if (node.right != null) {
                    node.right!!.next = node?.next?.left
                }
            }

            connectNext(node.left)
            connectNext(node.right)
        }

        connectNext(root)
        return root
    }

    fun connect3Plus(root: Node?): Node? {

        if (root == null) return root

        if (root.left != null) {
            root.left!!.next = root!!.right
            if (root.right != null) {
                root.right!!.next = root?.next?.left
            }
        }

        connect(root.left)
        connect(root.right)

        return root
    }
}

class Solution116_20220601 {

    class Node(var `val`: Int) {
        var left: Node? = null
        var right: Node? = null
        var next: Node? = null
    }

    fun conn(root: Node?) {
        if (root == null || root.left == null) return

        // root.left != null && root.right != null
        root.left!!.next = root.right!!
        root.right!!.next = root.next?.left

        conn(root.left!!)
        conn(root.right!!)
    }

    fun connect(root: Node?): Node? {
        conn(root)
        return root
    }
}

/// 462. 最少移动次数使数组元素相等 II
class Solution462 {
    /// 排序
    fun minMoves2(nums: IntArray): Int {
        if (nums.size <= 1) return 0

        nums.sort()

        var count = 0
        val target = nums[(0 + nums.size - 1) ushr 1]
        nums.forEach {
            count += Math.abs(it - target)
        }

        return count
    }

    /// 找中位数，快排分区思想
    fun minMoves2_MidValue(nums: IntArray): Int {
        if (nums.size <= 1) return 0

        /// 找到数组中第 N 小的数
        val rand = Random()
        fun findNth(start: Int, end: Int, nth: Int): Int {
            val pivotIndex = rand.nextInt(end - start + 1) + start
            nums[pivotIndex] = nums[end].also { nums[end] = nums[pivotIndex] }
            val pivot = nums[end]
            var i = start
            for (j in start until end) {
                if (nums[j] < pivot) {
                    if (i != j) {
                        nums[i] = nums[j].also { nums[j] = nums[i] }
                    }
                    i++
                }
            }
            if (i != end) {
                nums[i] = nums[end].also { nums[end] = nums[i] }
            }

            if (i - start + 1 == nth) {
                return nums[i]
            } else if (i - start + 1 > nth) {
                return findNth(start, i - 1, nth)
            } else { // i - start + 1 < nth
                return findNth(i + 1, end, nth - (i - start + 1))
            }
        }

        var count = 0
        val target = findNth(0, nums.size - 1, ((nums.size - 1) ushr 1) + 1)
        nums.forEach {
            count += Math.abs(it - target)
        }

        return count
    }
}

/// 542. 01 矩阵
class Solution542 {
    fun updateMatrix(mat: Array<IntArray>): Array<IntArray> {
        val rowMax = mat.size - 1
        val colMax = mat[0].size - 1

        fun get(r: Int, c: Int): Int {
            if (r < 0 || r > rowMax || c < 0 || c > colMax) {
                return Int.MAX_VALUE
            }
            return mat[r][c]
        }

        fun getMin(r: Int, c: Int) {
            var min = minOf(get(r - 1, c), get(r, c - 1))
            if (min != Int.MAX_VALUE) {
                min += 1
            }
            mat[r][c] = min
        }

        fun getMin2(r: Int, c: Int) {
            var min = minOf(get(r + 1, c), get(r, c + 1))
            if (min != Int.MAX_VALUE) {
                min += 1
            }
            mat[r][c] = minOf(mat[r][c], min)
        }

        for (r in 0 until rowMax + 1) {
            for (c in 0 until colMax + 1) {
                if (mat[r][c] != 0) {
                    getMin(r, c)
                }
            }
        }

        for (r in rowMax downTo 0) {
            for (c in colMax downTo 0) {
                if (mat[r][c] != 0) {
                    getMin2(r, c)
                }
            }
        }

        return mat
    }
}

/// 994. 腐烂的橘子
class Solution994 {
    fun orangesRotting_1(grid: Array<IntArray>): Int {
        var rowMax = grid.size - 1
        var colMax = grid[0].size - 1

        // 2 烂橘子。
        // 1 好橘子。
        // 0 神奇隔板，隔离腐烂和新鲜。

        fun get(r: Int, c: Int): Int {
            if (r < 0 || r > rowMax || c < 0 || c > colMax) {
                return Int.MAX_VALUE
            }
            if (grid[r][c] == Int.MIN_VALUE) {
                return Int.MAX_VALUE
            }
            return grid[r][c]
        }

        val indices = listOf(Pair(1, 0), Pair(-1, 0), Pair(0, 1), Pair(0, -1))

        // 计算每个橘子距离它最近的烂橘子的最小距离，考虑空格的阻挡(距离无限大)
        var good = (colMax + 1) * (rowMax + 1)

        var rotted = LinkedList<Pair<Int, Int>>()
        var seen = Array(rowMax + 1) { BooleanArray(colMax + 1) }
        var gridTmp = Array(rowMax + 1) { row ->
            IntArray(colMax + 1) { col ->
                if (grid[row][col] == 2) {
                    rotted.add(Pair(row, col))
                    seen[row][col] = true
                    good--
                } else if (grid[row][col] == 0) {
                    seen[row][col] = true
                    good--
                }
                0
            }
        }

        // 然后取最大值
        var max = 0
        while (rotted.isNotEmpty()) {
            val (row, col) = rotted.poll()
            for ((dr, dc) in indices) {
                val r = row + dr
                val c = col + dc
                if (r in 0..rowMax && c in 0..colMax && !seen[r][c]) {
                    gridTmp[r][c] = gridTmp[row][col] + 1
                    max = gridTmp[r][c]
                    rotted.offer(Pair(r, c))
                    seen[r][c] = true
                    good--
                }
            }
        }

        if (good > 0) return -1

        return max
    }

    fun orangesRotting(grid: Array<IntArray>): Int {
        var rowMax = grid.size - 1
        var colMax = grid[0].size - 1

        // 2 烂橘子。
        // 1 好橘子。
        // 0 神奇隔板，隔离腐烂和新鲜。

        val directions = listOf(Pair(1, 0), Pair(-1, 0), Pair(0, 1), Pair(0, -1))

        // 计算每个橘子距离它最近的烂橘子的最小距离，考虑空格的阻挡(距离无限大)
        var good = 0

        var rotten = LinkedList<Triple<Int, Int, Int>>()
        for (row in 0 until rowMax + 1) {
            for (col in 0 until colMax + 1) {
                if (grid[row][col] == 2) {
                    rotten.offer(Triple(row, col, 0))
                    grid[row][col] = 0 // 用 0 代表无法搜索的位置
                } else if (grid[row][col] == 1) {
                    // 用 1 代表可以搜索的位置（是好橘子）
                    good++
                }
            }
        }

        // 然后取最大值
        var maxDepth = 0
        while (rotten.isNotEmpty()) {
            val (row, col, depth) = rotten.poll()
            maxDepth = depth
            for ((dr, dc) in directions) {
                val r = row + dr
                val c = col + dc
                if (r in 0..rowMax && c in 0..colMax && grid[r][c] == 1) { // 1 说明原来是好的
                    good--         // 搞坏一个橘子
                    grid[r][c] = 0 // 并标记它不可以再次被搜索到
                    rotten.offer(Triple(r, c, depth + 1)) // 将当前位置加入带搜索队列（它已经准备好去搞坏其它橘子）
                }
            }
        }

        if (good > 0) return -1
        return maxDepth
    }
}

/// 56. 合并区间
class Solution56 {
    fun merge(intervals: Array<IntArray>): Array<IntArray> {
        if (intervals.isEmpty()) return intervals

        intervals.sortBy { it.first() }

        var results = mutableListOf<IntArray>()

        var i = 1
        var intervalList = intervals[0]
        while (i < intervals.size) {
            val currentList = intervals[i]
            if (currentList.first()!! > intervalList.last()!!) {
                results.add(intervalList)
                intervalList = currentList
            } else { // currentList.first()!! <= intervalList.last()!!
                intervalList[1] = maxOf(currentList[1], intervalList[1])
            }
            i++
        }
        results.add(intervalList)

        return results.toTypedArray()
    }
}

/// 961. 在长度 2N 的数组中找出重复 N 次的元素
class Solution961 {
    fun repeatedNTimes_1(nums: IntArray): Int {
        var seen = mutableMapOf<Int, Boolean>()
        for (n in nums) {
            if (seen.containsKey(n)) {
                return n
            }
            seen[n] = true
        }
        return -1
    }

    fun repeatedNTimes(nums: IntArray): Int {
        nums.sort()
        for (i in 1 until nums.size) {
            if (nums[i] == nums[i - 1]) {
                return nums[i]
            }
        }
        return -1
    }
}

/// 查找数组中重复次数最多的数字
/*
给定一个大小为n的数组，该数组包含数字的范围在 [0...k-1]， k是一个正整数,k < = n。在这个数组找到重复次数最多的数字。
要求时间复杂度为n，空间复杂度为1，可以使用原数组。
 */
class Solution20220521 {
    fun findMostCount(nums: IntArray, k: Int): Int {
        var maxCount = 0
        var max = 0
        for (n in nums) {
            nums[n % k] += k // 通过 n % k 来找到 n 在 nums 中的次数缓存位置，+k
            if (maxCount < nums[n % k]) {
                maxCount = nums[n % k]
                max = n % k
            }
        }
        return max
    }
}

/// 39. 组合总和
class Solution39 {
    fun combinationSum1(candidates: IntArray, target: Int): List<List<Int>> {
        fun searchLessThanOrEqual(): Int {
            var l = 0
            var r = candidates.size - 1

            while (l <= r) {
                val m = (l + r) ushr 1
                if (candidates[m] <= target && (m == candidates.size - 1 || candidates[m + 1] > target)) {
                    return m
                } else if (candidates[m] <= target) {
                    l = m + 1
                } else { // target < candidates[m]
                    r = m - 1
                }
            }

            return -1
        }

        var results = mutableSetOf<List<Int>>()
        candidates.sort()

        val to = searchLessThanOrEqual()
        if (to == -1) {
            return results.toList()
        }

        fun search(target: Int, selected: List<Int>) {
            for (i in 0 until to + 1) {
                val candidate = candidates[i]
                if (target == candidate) {
                    val nums = selected.toMutableList()
                    nums.add(candidate)
                    nums.sort()
                    results.add(nums)
                    // candidate 继续增长无法满足 target，因为总会比 target 大，所以直接退出循环
                    // candidates 中不包含重复的数
                    break
                } else if (candidate > target) {
                    break // candidate 比 target 大，所以直接退出循环
                } else { // candidate < target
                    val nums = selected.toMutableList()
                    nums.add(candidate)
                    nums.sort()
                    search(target - candidate, nums)
                }
            }
        }

        search(target, listOf<Int>())
        return results.toList()
    }

    class Solution {
        fun combinationSum(candidates: IntArray, target: Int): List<List<Int>> {
            val result = ArrayList<ArrayList<Int>>()

            if (candidates.size == 0) return emptyList()

            val path = ArrayDeque<Int>()
            dfs(candidates, target, 0, candidates.size - 1, path, result)
            return result
        }

        fun dfs(
            candidates: IntArray,
            target: Int,
            from: Int,
            to: Int,
            path: ArrayDeque<Int>,
            result: ArrayList<ArrayList<Int>>
        ) {
            for (i in from until to + 1) {
                val candidate = candidates[i]
                if (candidate == target) {
                    path.addLast(candidate)
                    result.add(ArrayList(path))
                    path.removeLast()
                } else if (candidate < target) {
                    path.addLast(candidate)
                    // 这里 dfs 时的关键点： from 的值为 i，即每次只去尝试当前候选数字以及排在它后面的数字，这样可以避免重复
                    dfs(candidates, target - candidate, i, to, path, result)
                    path.removeLast()
                } else { // candidate > target
                    continue
                }
            }
        }
    }
}

/// 39. 组合总和
class Solution39_20220604 {
    fun combinationSum(candidates: IntArray, target: Int): List<List<Int>> {
        val results = mutableListOf<List<Int>>()
        fun dfs(selection: LinkedList<Int>, sum: Int, index: Int) {
            if (sum > target) {
                return
            }
            if (sum == target) {
                results.add(selection.toList())
                return
            }
            for (i in index until candidates.size) {
                selection.offerLast(candidates[i])
                dfs(selection, sum + candidates[i], i)
                selection.pollLast()
            }
        }

        dfs(LinkedList<Int>(), 0, 0)
        return results
    }
}

/// 77. 组合
class Solution77 {
    fun dfs(
        from: Int,
        to: Int,
        path: ArrayDeque<Int>,
        targetLength: Int,
        result: MutableList<List<Int>>
    ) {
        if (path.size == targetLength) {
            result.add(ArrayList(path))
            return
        }

        for (i in from until to + 1) {
            path.addLast(i)
            dfs(i + 1, to, path, targetLength, result)
            path.removeLast()
        }
    }

    fun combine(n: Int, k: Int): List<List<Int>> {
        var result = mutableListOf<List<Int>>()
        dfs(1, n, ArrayDeque<Int>(), k, result)
        return result
    }
}

/// 46. 全排列
class Solution46 {
    fun dfs(nums: MutableList<Int>, path: ArrayDeque<Int>, result: MutableList<List<Int>>) {
        if (nums.isEmpty()) {
            result.add(ArrayList(path))
            return
        }

        for (i in 0 until nums.size) {
            val num = nums[i]
            path.addLast(num)
            nums.removeAt(i)
            dfs(nums, path, result)
            nums.add(i, num)
            path.removeLast()
        }
    }

    fun permute(nums: IntArray): List<List<Int>> {
        var result = mutableListOf<List<Int>>()
        dfs(nums.toMutableList(), ArrayDeque<Int>(), result)
        return result
    }
}

class Solution46_20220604 {
    fun permute(nums: IntArray): List<List<Int>> {
        val results = mutableListOf<List<Int>>()
        val used = BooleanArray(nums.size)

        fun dfs(path: LinkedList<Int>) {
            if (path.size == nums.size) {
                results.add(path.toList())
                return
            }
            for (i in nums.indices) {
                if (!used[i]) {
                    val num = nums[i]
                    path.offerLast(num)
                    used[i] = true
                    dfs(path)
                    used[i] = false
                    path.pollLast()
                }
            }
        }

        dfs(LinkedList<Int>())
        return results
    }

    fun permute_NoUsed(nums: IntArray): List<List<Int>> {
        val results = mutableListOf<List<Int>>()
        fun dfs(index: Int) {
            if (index == nums.size) {
                results.add(nums.toList())
                return
            }
            for (i in index until nums.size) {
                nums[index] = nums[i].also { nums[i] = nums[index] }
                dfs(index + 1)
                nums[index] = nums[i].also { nums[i] = nums[index] }
            }
        }

        dfs(0)
        return results
    }
}

class Solution46_1 {
    val result = mutableListOf<List<Int>>()
    fun dfs(nums: IntArray, path: ArrayDeque<Int>) {
        if (path.size >= nums.size) {
            result.add(ArrayList(path))
            return
        }

        for (num in nums) {
            if (!path.contains(num)) {
                path.addLast(num)
                dfs(nums, path)
                path.removeLast()
            }
        }
    }

    fun permute(nums: IntArray): List<List<Int>> {
        result.clear()
        dfs(nums, ArrayDeque<Int>())
        return result
    }
}

/// 784. 字母大小写全排列
class Solution784 {
    val result = mutableListOf<String>()
    fun dfs(s: String, from: Int, path: StringBuffer) {
        if (from == s.length) {
            result.add(path.toString())
            return
        }

        var pathLength = path.length
        for (i in from until s.length) {
            val c = s[i]
            if (c in 'a'..'z' || c in 'A'..'Z') {
                path.append(c.toLowerCase())
                dfs(s, i + 1, path)
                path.deleteCharAt(path.length - 1)

                path.append(c.toUpperCase())
                dfs(s, i + 1, path)
                path.deleteCharAt(path.length - 1)

                path.delete(pathLength, path.length)

                return
            } else {
                path.append(c)
            }
        }

        result.add(path.toString())
        path.delete(pathLength, path.length)
    }

    fun letterCasePermutation(s: String): List<String> {
        result.clear()
        val path = StringBuffer()
        dfs(s, 0, path)
        return result
    }
}

/// 剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
class SolutionJZOffer21 {
    fun exchange1(nums: IntArray): IntArray {
        var l = 0
        var r = nums.size - 1
        // 目标： 左边奇数，右边偶数
        while (l < r) {
            while (l <= r && nums[l] % 2 == 1) {
                l++
            }
            while (l <= r && nums[r] % 2 == 0) {
                r--
            }
            if (l < r) {
                nums[l] = nums[r].also { nums[r] = nums[l] }
                l++
                r--
            }
        }
        return nums
    }

    /// 知识点判断整数奇偶，通过 and 1 判断
    /// n and 1 == 1 => 奇数
    /// n and 1 == 0 => 偶数
    fun exchange(nums: IntArray): IntArray {
        var l = 0
        var r = nums.size - 1
        // 目标： 左边奇数，右边偶数
        while (l < r) {
            while (l <= r && nums[l] and 1 == 1) {
                l++
            }
            while (l <= r && nums[r] and 1 == 0) {
                r--
            }
            if (l < r) {
                nums[l] = nums[r].also { nums[r] = nums[l] }
                l++
                r--
            }
        }
        return nums
    }
}

/// 1931. 用三种不同颜色为网格涂色
class Solution1931 {
    fun colorTheGrid(m: Int, n: Int): Int {
        val mod = 1000000007

        // mxn，m 行，n 列。 因为题意限制 m 属于更小的数。
        // 一行 m 个格子
        // 一列 n 个格子

        val validRows = mutableMapOf<Int, IntArray>()

        // 在 [0, 3^m) 范围内美剧满足要求的 mask
        // 用 0代表红色，1代表绿色，2代表蓝色
        val maskMax = Math.pow(3.0, m.toDouble()).toInt()
        for (mask in 0 until maskMax) {
            val colors = IntArray(m)
            var tmpMask = mask
            for (i in 0 until m) {
                colors[i] = tmpMask % 3
                tmpMask /= 3
            }
            var isValid = true
            for (i in 0 until m - 1) {
                if (colors[i] == colors[i + 1]) {
                    isValid = false
                    break
                }
            }
            if (isValid) {
                validRows[mask] = colors
            }
        }

        // 预处理所有的 (mask1, mask2) 二元组，满足 mask1 和 mask2 相邻时，同一列上相邻两格的颜色不同
        val validRowRows = mutableMapOf<Int, MutableList<Int>>()
        for ((mask1, colors1) in validRows) {
            for ((mask2, colors2) in validRows) {
                var isValid = true
                for (i in 0 until m) {
                    if (colors1[i] == colors2[i]) {
                        isValid = false
                        break
                    }
                }
                if (isValid) {
                    if (!validRowRows.containsKey(mask1)) {
                        validRowRows[mask1] = mutableListOf(mask2)
                    } else {
                        validRowRows[mask1]!!.add(mask2)
                    }
                }
            }
        }

        // count[i][mask] 表示第 0 到 i 行已经填上符合要求的颜色，且第 i 行的颜色与状态 mask 对应的情况下，总共的涂色方式数。
        // count[i][mask] = count[i-1][mask'] {all i-1 row mask' valid with mask} ==> 由于 count[i][mask] 仅和
        // count[i-1][mask'] 相关，可以仅用一维数组即可。
        // 显然 count[0][mask] = 1
        // 结果即为所有 mask 对应 count[n-1][mask] 的加和。

        var count = IntArray(maskMax)
        for ((mask, _) in validRows) {
            // dp[0][mask] = 1
            count[mask] = 1
        }

        for (i in 1 until n) {
            val tmpCount = IntArray(maskMax)
            for ((maskCurrentRow, _) in validRows) {
                if (!validRowRows.containsKey(maskCurrentRow)) continue
                for (maskPreviousRow in validRowRows[maskCurrentRow]!!) {
                    tmpCount[maskCurrentRow] += count[maskPreviousRow]
                    if (tmpCount[maskCurrentRow] >= mod) {
                        tmpCount[maskCurrentRow] -= mod
                    }
                }
            }
            count = tmpCount
        }

        var totalCount = 0
        count.forEach {
            totalCount += it
            if (totalCount >= mod) {
                totalCount -= mod
            }
        }
        return totalCount
    }
}

/// 120. 三角形最小路径和
class Solution120 {
    fun minimumTotal1(triangle: List<List<Int>>): Int {
        // sum[i] = min(sum[i], sum[i - 1])
        val sum = IntArray(triangle.last()!!.size + 1) { Int.MAX_VALUE }
        sum[0] = triangle[0]!![0]!!
        for (i in 1 until triangle.size) {
            val nums = triangle[i]!!
            var sumJ_1 = sum[0] // 保存一下 sum[j] 的状态
            sum[0] += nums[0]
            for (j in 1 until nums.size) {
                var sumJ_1Tmp = sum[j] // 保存一下 sum[j] 的状态, 因为下一行代码会将 sum[j] 更新
                sum[j] = minOf(sum[j], sumJ_1) + nums[j]
                sumJ_1 = sumJ_1Tmp // 接住 sum[j] 的值，下一次遍历时使用
            }
        }

        var min = sum[0]
        for (i in 1 until sum.size) {
            if (sum[i] < min) {
                min = sum[i]
            }
        }
        return min
    }

    fun minimumTotal(triangle: List<List<Int>>): Int {
        // sum[i] = min(sum[i], sum[i - 1])
        val sum = IntArray(triangle.last()!!.size + 1) { Int.MAX_VALUE }
        sum[0] = triangle[0]!![0]!!
        for (i in 1 until triangle.size) {
            val nums = triangle[i]!!
            for (j in nums.size - 1 downTo 1) {
                sum[j] = minOf(sum[j], sum[j - 1]) + nums[j]
            }
            sum[0] += nums[0]
        }

        var min = sum[0]
        for (i in 1 until sum.size) {
            if (sum[i] < min) {
                min = sum[i]
            }
        }
        return min
    }
}

/// 128. 最长连续序列
class Solution128 {
    fun longestConsecutive(nums: IntArray): Int {
        val numsSet = mutableSetOf<Int>()
        for (n in nums) {
            numsSet.add(n)
        }

        var maxLength = 0
        for (n in nums) {
            if (!numsSet.contains(n - 1)) {
                var currentNum = n
                var currentLength = 1
                while (numsSet.contains(currentNum + 1)) {
                    currentLength += 1
                    currentNum += 1
                }
                maxLength = maxOf(maxLength, currentLength)
            }
        }

        return maxLength
    }

    fun longestConsecutive1(nums: IntArray): Int {
        var max = 0
        val indexNumbers = mutableMapOf<Int, LinkedList<Int>>()
        for (n in nums) {
            if (indexNumbers.containsKey(n)) {
                continue
            }

            val hasNMinus1 = indexNumbers.containsKey(n - 1)
            val hasNPlus1 = indexNumbers.containsKey(n + 1)
            if (hasNMinus1 && hasNPlus1) {
                val numbersNMinus1 = indexNumbers[n - 1]!!
                numbersNMinus1.addLast(n)
                indexNumbers[n] = numbersNMinus1

                val numbersNPlus1 = indexNumbers[n + 1]!!
                numbersNMinus1.addAll(numbersNPlus1)
                numbersNMinus1.forEach { indexNumbers[it] = numbersNMinus1 }

                max = maxOf(max, numbersNMinus1.size)
            } else if (hasNMinus1) {
                val numbersNMinus1 = indexNumbers[n - 1]!!
                numbersNMinus1.addLast(n)
                indexNumbers[n] = numbersNMinus1

                max = maxOf(max, numbersNMinus1.size)
            } else if (hasNPlus1) {
                val numbersNPlus1 = indexNumbers[n + 1]!!
                numbersNPlus1.addFirst(n)
                indexNumbers[n] = numbersNPlus1

                max = maxOf(max, numbersNPlus1.size)
            } else {
                indexNumbers[n] = LinkedList(listOf(n))

                max = maxOf(max, 1)
            }
        }

        return max
    }
}


/// 675. 为高尔夫比赛砍树
class Solution675 {
    fun cutOffTree(forest: List<List<Int>>): Int {
        // forest[i][j] = 0阻隔，1地面（可通行），>1树木（可通行，树木高度）
        // distance[i][j] 代表从 (0, 0) 走到 (i, j) 的距离
        val rowCount = forest.size
        val colCount = forest[0]!!.size
        val directions = listOf(Pair(1, 0), Pair(-1, 0), Pair(0, 1), Pair(0, -1))

        fun search(fromRow: Int, fromCol: Int, toRow: Int, toCol: Int): Int {
            if (fromRow == toRow && fromCol == toCol) return 0

            val tmpForest = forest.map { it.toIntArray() }
            val willSearchPoints = LinkedList(listOf(Triple(fromRow, fromCol, 0)))
            while (willSearchPoints.isNotEmpty()) {
                val (row, col, step) = willSearchPoints.pollFirst()
                for (direction in directions) {
                    val r = row + direction.first
                    val c = col + direction.second
                    if (r in 0 until rowCount && c in 0 until colCount) {
                        val height = tmpForest[r][c]
                        if (height > 0) {
                            if (r == toRow && c == toCol) {
                                return step + 1
                            }
                            tmpForest[r][c] = 0
                            willSearchPoints.offerLast(Triple(r, c, step + 1))
                        }
                    }
                }
            }

            return -1 // 不可达
        }


        var trees = PriorityQueue<Triple<Int, Int, Int>>() { (_, _, h1), (_, _, h2) -> h1 - h2 }
        for (row in 0 until rowCount) {
            for (col in 0 until colCount) {
                val height = forest[row]!![col]!!
                if (height > 1) {
                    trees.add(Triple(row, col, height))
                }
            }
        }

        var (preRow, preCol, _) = trees.poll()
        var minDistance = search(0, 0, preRow, preCol)
        if (minDistance == -1) return -1

        while (trees.isNotEmpty()) {
            val (row, col, _) = trees.poll()
            val distance = search(preRow, preCol, row, col)
            if (distance == -1) return -1
            minDistance += distance
            preRow = row
            preCol = col
        }

        return minDistance
    }
}

/// 23. 合并K个升序链表
class Solution23 {
    fun mergeKLists(lists: Array<ListNode?>): ListNode? {
        val prehead = ListNode(0)
        var current = prehead

        val queue = PriorityQueue<ListNode> { a, b -> a.`val` - b.`val` }
        lists.forEach {
            if (it != null) queue.add(it!!)
        }

        while (queue.isNotEmpty()) {
            val node = queue.poll()
            if (node.next != null) {
                queue.add(node.next!!)
                node.next = null
            }

            current.next = node
            current = node
        }

        return prehead.next
    }
}

/// 965. 单值二叉树
class Solution965 {
    class TreeNode(var `val`: Int) {
        var left: TreeNode? = null
        var right: TreeNode? = null
    }

    fun isUnivalTree(root: TreeNode?): Boolean {
        if (root == null) return true

        return (root.left == null || root.left!!.`val` == root.`val`) && (root.right == null || root.right!!.`val` == root.`val`) && isUnivalTree(
            root.left
        ) && isUnivalTree(root.right)
    }
}

/// 231. 2 的幂
class Solution231 {
    fun isPowerOfTwo(n: Int): Boolean {
        // n & (n - 1) == 0  => n & (n - 1) 可以将 n 最低位的 1 移除
        return n > 0 && (n and (n - 1)) == 0
        // n & -n == n
        //return n > 0 && (n and -n) == n
        // n 为 2^30 的约数 (Int.SIZE_BITS-2 == 30)
        //return n > 0 && (1 shl 30) % n == 0
    }

    fun isPowerOfTwo_For(n: Int): Boolean {
        if (n <= 0) return false

        var bitCount = 0
        for (i in 0 until Int.SIZE_BITS) {
            if ((n and (1 shl i)) > 0) {
                bitCount++
                if (bitCount > 1) {
                    return false
                }
            }
        }
        return bitCount == 1
    }

    fun isPowerOfTwo_While(n: Int): Boolean {
        if (n == 0) return false // 特殊判断, 0 非2的幂次方
        if (n == 1) return true // 特殊情况判断, 2^0 等于 1
        if ((n and 1) == 1) return false // 特殊情况判断, 除1之外的奇数不可能为2的幂次方
        var value = n
        while (((value shr 1) shl 1) == value) {
            value = value shr 1
        }
        return value == 1
    }
}

/// 191. 位1的个数
class Solution191 {
    // you need treat n as an unsigned value
    fun hammingWeight(n: Int): Int {
        var count = 0
        for (i in 0 until Int.SIZE_BITS) {
            if (n and (1 shl i) != 0) {
                count++
            }
        }
        return count
    }

    // you need treat n as an unsigned value
    fun hammingWeight_NandN_1(n: Int): Int {
        var count = 0
        var n = n
        while (n != 0) {
            n = n and (n - 1)
            count++
        }
        n.countOneBits()
        return count
    }
}

/// 407. 接雨水 II
class Solution407 {
    fun trapRainWater(walls: Array<IntArray>): Int {
        val rowCount = walls.size
        val colCount = walls[0].size

        if (rowCount <= 2 || colCount <= 2) return 0

        // 是否被访问过
        val visited = Array(rowCount) { BooleanArray(colCount) }
        val wallQueue = PriorityQueue<Triple<Int, Int, Int>> { h1, h2 -> h1.third - h2.third }

        // 遍历方块四周
        for (row in listOf(0, rowCount - 1)) {
            for (col in 0 until colCount) {
                visited[row][col] = true
                wallQueue.offer(Triple(row, col, walls[row][col]))
            }
        }
        for (col in listOf(0, colCount - 1)) {
            for (row in 1 until rowCount - 1) {
                visited[row][col] = true
                wallQueue.offer(Triple(row, col, walls[row][col]))
            }
        }

        var waterTotal = 0
        val directions = listOf(Pair(1, 0), Pair(-1, 0), Pair(0, 1), Pair(0, -1))
        while (wallQueue.isNotEmpty()) {
            val (row, col, height) = wallQueue.poll()
            directions.forEach { (dr, dc) ->
                val r = row + dr
                var c = col + dc
                if (r in 0 until rowCount && c in 0 until colCount && !visited[r][c]) {
                    if (height > walls[r][c]) {
                        /*
                        (1) 假设装满水后的墙(可能包含水)的高度为 water[r][c], 原墙高度为 walls[r][c], 则水体的单位
                        个数为 water[r][c] - walls[r][c].
                        (2) 显然, 整个墙体的四周的墙是不可能蓄水的, 所以四周每个墙的 water[r][c] == walls[r][c],
                        而且它们构成一个蓄水的联通墙体, 根据木桶原理, 我们可以从最矮的墙开始进行宽度优先搜索(上下左右),
                        根据 (1) 的设定, 可以确定被搜索到的墙的蓄水量, 即 maxOf(0, 最矮墙蓄水后高度 - 当前遍历的墙的高度),
                        根据以上流程继续遍历, 直到所有的墙都被扫描一遍.
                         */
                        waterTotal += height - walls[r][c]
                    }
                    wallQueue.offer(Triple(r, c, maxOf(height, walls[r][c])))
                    visited[r][c] = true
                }
            }
        }

        return waterTotal
    }

    fun trapRainWater_BFS(walls: Array<IntArray>): Int {
        val rowCount = walls.size
        val colCount = walls[0].size

        if (rowCount <= 2 || colCount <= 2) return 0

        var highest = 0 // 最高墙高度
        for (row in 0 until rowCount) {
            for (col in 0 until colCount) {
                highest = maxOf(highest, walls[row][col])
            }
        }

        /*
        假设来个一个魔法师能让所有柱子凭空装满到达 highest 高度的水, 然后再从外到内一个柱子一个柱子解除魔法.
        可以知道, 四周的墙上的水最终都会流失, 我们可以从 walls[0][0] 开始, 将其四周的墙上的水
        都下降到 maxOf(它本身墙高度, walls[0][0]), 下降过程中记录流失的水.
        最终总流失水量 + 所有墙高度总和 + 被留下来的水 = m * n * highest
        runAwayWaterTotal + allTotal
        */
        val waters = Array(rowCount) { IntArray(colCount) { highest } }
        var willSearchWalls = LinkedList<Pair<Int, Int>>()
        for (row in 0 until rowCount) {
            for (col in 0 until colCount) {
                if ((row == 0 || row == rowCount - 1 || col == 0 || col == colCount - 1) && waters[row][col] > walls[row][col]) {
                    willSearchWalls.offer(Pair(row, col))
                    waters[row][col] = walls[row][col]
                }
            }
        }

        val directions = listOf(Pair(1, 0), Pair(-1, 0), Pair(0, 1), Pair(0, -1))
        while (willSearchWalls.isNotEmpty()) {
            val (row, col) = willSearchWalls.poll()
            for ((dr, dc) in directions) {
                val r = row + dr
                val c = col + dc

                if (r in 0 until rowCount && c in 0 until colCount && (waters[row][col] < waters[r][c] && waters[r][c] > walls[r][c])) {
                    waters[r][c] = maxOf(walls[r][c], waters[row][col])
                    willSearchWalls.offer(Pair(r, c))
                }
            }
        }

        var res = 0
        for (row in 0 until rowCount) {
            for (col in 0 until colCount) {
                res += waters[row][col] - walls[row][col]
            }
        }

        return res
    }
}

/// 190. 颠倒二进制位
class Solution190 {
    fun reverseEvery2Bits(i: Int): Int {
        // 0b10101010101010101010101010101010
        // 0b01010101010101010101010101010101 0x55555555
        return ((i and 0x55555555) shl 1) or ((i ushr 1) and 0x55555555)
    }

    fun reverseEvery4Bits(i: Int): Int {
        // 0b11001100110011001100110011001100
        // 0b00110011001100110011001100110011 0x33333333
        return ((i and 0x33333333) shl 2) or ((i ushr 2) and 0x33333333)
    }

    fun reverseEvery8Bits(i: Int): Int {
        // 0b11110000111100001111000011110000
        // 0b00001111000011110000111100001111 0x0F0F0F0F
        return ((i and 0x0F0F0F0F) shl 4) or ((i ushr 4) and 0x0F0F0F0F)
    }

    fun reverseBytes(i: Int): Int {
        return (i ushr 24) or (i shl 24) or ((i ushr 8) and 0xFF00) or ((i and 0xFF00) shl 8)
    }

    fun reverseBits(n: Int): Int {
        return reverseBytes(reverseEvery8Bits(reverseEvery4Bits(reverseEvery2Bits(n))))
    }

    ////// 有符号右移, 高位补符号位, 低位丢弃; 无符号右移, 高位补0, 低位丢弃
    ////// 左移没有有符号与无符号区分 (why? 定义如此), 高位丢弃, 低位补0

    // you need treat n as an unsigned value
    fun reverseBits2(n: Int): Int {
        var n = n
        var res = 0
        for (i in 0 until Int.SIZE_BITS) {
            if (n and 1 == 1) {
                res = res or (1 shl (Int.SIZE_BITS - i - 1))
            }
            n = n ushr 1
        }
        return res
    }

    fun reverseBits2Better(n: Int): Int {
        var res = 0
        for (i in 0 until Int.SIZE_BITS) {
            if (n and (1 shl i) != 0) {
                res = res or (1 shl (Int.SIZE_BITS - i - 1))
            }
        }
        return res
    }

    // you need treat n as an unsigned value
    fun reverseBits1(n: Int): Int {
        var n = n
        var low = 1
        var high = Int.SIZE_BITS - 2
        while (low < high) {
            val lowBitIs1 = n and (1 shl low) != 0
            val highBitIs1 = (n and (1 shl high)) != 0
            // update lowBit
            if (highBitIs1) {
                n = (n or (1 shl low))
            } else {
                n = (n or (1 shl low).inv())
            }
            n.toString(2)
            // update highBit
            if (lowBitIs1) {
                n = (n or (1 shl high))
            } else {
                n = (n or (1 shl high).inv())
            }

            low++
            high--
        }
        return n
    }

    /// 双指针, 高低位互换

    // you need treat n as an unsigned value
    fun reverseBits_TwoPointers(n: Int): Int {
        var n = n
        var low = 0
        var high = Int.SIZE_BITS - 1
        while (low < high) {
            val lowBitIs1 = (n and (1 shl low)) != 0
            val highBitIs1 = (n and (1 shl high)) != 0

            // low bit
            if (highBitIs1) {
                // 低位设置为1
                n = n or (1 shl low)
            } else {
                // 低位设置为0
                n = n and (1 shl low).inv()
            }

            // high bit
            if (lowBitIs1) {
                // 高位设置为1
                n = n or (1 shl high)
            } else {
                // 高位设置为0
                n = n and (1 shl high).inv()
            }

            low++
            high--
        }
        return n
    }

    //// byte flip 2,4,8,16,32

    // you need treat n as an unsigned value
    fun reverseBits_BITS(n: Int): Int {
        var i = n
        // 0x11111111111111111111111111111111

        // 0x01010101010101010101010101010101 0x55555555
        i = ((i and 0x55555555) shl 1) or ((i ushr 1) and 0x55555555)

        // 0x00110011001100110011001100110011 0x33333333
        i = ((i and 0x33333333) shl 2) or ((i ushr 2) and 0x33333333)

        // 0x00001111000011110000111100001111 0x0f0f0f0f
        i = ((i and 0x0f0f0f0f) shl 4) or ((i ushr 4) and 0x0f0f0f0f)

        i = (i ushr 24) or (i shl 24) or ((i and 0xff00) shl 8) or ((i ushr 8) and 0xff00)

        return i
    }
}

/// 467. 环绕字符串中唯一的子字符串
class Solution467 {
    /// 暴力搜索, 超时.
    fun findSubstringInWraproundString_0(p: String): Int {
        // 找到 p 的所有非空子串, 逐一检查是否属于 s
        fun nextChar(c: Char): Char {
            if (c == 'z') return 'a'
            return c + 1
        }

        var mem = mutableMapOf<String, Boolean>()
        fun check(p: String): Boolean {
            if (mem.containsKey(p)) return mem[p]!!

            if (p.length == 1) return true.also { mem[p] = it }
            if (p.length == 2) return (nextChar(p[0]) == p[1]).also { mem[p] = it }
            return ((nextChar(p[0]) == p[1]) && check(p.substring(1))).also { mem[p] = it }
        }

        var allSubs = mutableSetOf<String>()
        for (length in 1..p.length) {
            for (start in 0..(p.length - length)) {
                allSubs.add(p.substring(start, start + length))
            }
        }

        var count = 0
        for (sub in allSubs) {
            if (check(sub)) count++
        }
        return count
    }

    /////

    fun findSubstringInWraproundString(p: String): Int {
        // dp[0] => dp['a']
        // dp[25] => dp['z']
        // dp[c]代表p的连续子串中以字符c结尾的个数.
        val dp = IntArray(26)
        var k = 1
        dp[p[0] - 'a'] = 1
        for (i in 1 until p.length) {
            val cur = p[i]
            val pre = p[i - 1]
            if (pre + 1 == cur || (pre == 'z' && cur == 'a')) {
                k++
            } else {
                k = 1
            }
            dp[cur - 'a'] = maxOf(dp[cur - 'a'], k)
        }
        return dp.sum()
    }
}

/// 55. 跳跃游戏
class Solution55 {
    fun canJump(nums: IntArray): Boolean {
        val n = nums.size
        var maxReachableIndex = 0

        for (i in nums.indices) {
            if (maxReachableIndex >= i) {
                maxReachableIndex = maxOf(maxReachableIndex, i + nums[i])
            }

            if (maxReachableIndex < i || maxReachableIndex >= n - 1) {
                break
            }
        }

        return maxReachableIndex >= n - 1
    }
}

class Solution55_20220606 {
    fun canJump(nums: IntArray): Boolean {
        val n = nums.size
        var maxReachable = nums[0]
        var i = 1
        while (i <= minOf(n - 1, maxReachable)) {
            maxReachable = maxOf(maxReachable, i + nums[i])
            if (maxReachable >= n - 1) return true
            i++
        }
        return maxReachable >= n - 1
    }
}

/// 48. 旋转图像
class Solution48 {
    /// 顺时针旋转, 以顶边为暂存, 依次 顶=>右, 右=>下, 下=>左
    fun rotate(matrix: Array<IntArray>): Unit {
        val n = matrix.size

        var start = 0
        var end = n - 1

        while (start < end) {
            // 顶边 -> 右边, 右边存放到顶边
            for (i in start until end) {
                matrix[start][i] = matrix[i][end].also { matrix[i][end] = matrix[start][i] }
            }
            // 右边 -> 底边, 底边存放到顶边
            for (i in start until end) {
                matrix[start][i] = matrix[end][end - (i - start)].also {
                    matrix[end][end - (i - start)] = matrix[start][i]
                }
            }
            // 底边 -> 左边, 左边 -> 顶边
            for (i in start until end) {
                matrix[start][i] = matrix[end - (i - start)][start].also {
                    matrix[end - (i - start)][start] = matrix[start][i]
                }
            }
            start++
            end--
        }
    }
}

/// 34. 在排序数组中查找元素的第一个和最后一个位置
class Solution34 {
    fun searchFirst(nums: IntArray, target: Int, from: Int = 0, to: Int = nums.size - 1): Int {
        var left = from
        var right = to
        while (left <= right) {
            val mid = (left + right) ushr 1
            if (nums[mid] == target && (mid == from || nums[mid - 1] != target)) {
                return mid
            } else if (target <= nums[mid]) {
                right = mid - 1
            } else { // target > nums[mid]
                left = mid + 1
            }
        }
        return -1
    }

    fun searchLast(nums: IntArray, target: Int, from: Int = 0, to: Int = nums.size - 1): Int {
        var left = from
        var right = to

        while (left <= right) {
            val mid = (left + right) ushr 1
            if (nums[mid] == target && (mid == to || nums[mid + 1] != target)) {
                return mid
            } else if (nums[mid] <= target) {
                left = mid + 1
            } else { // nums[mid] > target
                right = mid - 1
            }
        }
        return -1
    }

    fun searchRange(nums: IntArray, target: Int): IntArray {
        return intArrayOf(searchFirst(nums, target), searchLast(nums, target))
    }
}

/// 34. 在排序数组中查找元素的第一个和最后一个位置
class Solution34_20220608 {
    fun bsfirst(nums: IntArray, target: Int): Int {
        var l = 0
        var r = nums.size - 1
        while (l <= r) {
            val m = (l + r) ushr 1
            if (nums[m] == target && (m == 0 || nums[m - 1] != target)) return m
            else if (target <= nums[m]) {
                r = m - 1
            } else {
                l = m + 1
            }
        }
        return -1
    }

    fun bslast(nums: IntArray, target: Int): Int {
        var l = 0
        var r = nums.size - 1
        while (l <= r) {
            val m = (l + r) ushr 1
            if (nums[m] == target && (m == nums.size - 1 || nums[m + 1] != target)) return m
            else if (target < nums[m]) {
                r = m - 1
            } else {
                l = m + 1
            }
        }
        return -1
    }

    fun searchRange(nums: IntArray, target: Int): IntArray {
        return intArrayOf(bsfirst(nums, target), bslast(nums, target))
    }
}

/// 33. 搜索旋转排序数组
class Solution33 {
    fun bsearch(nums: IntArray, target: Int, from: Int = 0, to: Int = nums.size - 1): Int {
        var left = from
        var right = to
        while (left < right) {
            val mid = (left + right) ushr 1
            if (nums[mid] < target) {
                left = mid + 1
            } else {
                right = mid
            }
        }

        if (left <= to && nums[left] == target) {
            return left
        }
        return -1
    }

    fun search(nums: IntArray, target: Int): Int {
        // 每次找 mid, mid可以将数组分成一个 "排序数组" 和一个 "旋转排序数组", 根据数组首尾可以判断区分.
        var left = 0
        var right = nums.size - 1
        while (left <= right) {
            val mid = (left + right) ushr 1
            if (nums[left] <= nums[mid]) {
                // left..mid 为排序数组, mid+1..right 为旋转排序数组
                if (nums[left] <= target && target <= nums[mid]) {
                    return bsearch(nums, target, left, mid)
                } else {
                    left = mid + 1
                }
            } else { // nums[left] > nums[mid]
                // left..mid 为旋转排序数组, mid+1..right 为排序数组
                if (nums[mid + 1] <= target && target <= nums[right]) {
                    return bsearch(nums, target, mid + 1, right)
                } else {
                    right = mid
                }
            }
        }
        return -1
    }
}

class Solution33_1 {
    fun search(nums: IntArray, target: Int): Int {
        // 每次找 mid, mid可以将数组分成一个 "排序数组" 和一个 "旋转排序数组", 根据数组首尾可以判断区分.
        val first = nums[0]
        val last = nums[nums.size - 1]

        var left = 0
        var right = nums.size - 1
        while (left <= right) {
            val mid = (left + right) ushr 1
            if (nums[mid] == target) {
                return mid
            }
            if (first <= nums[mid]) {
                // left..mid 为排序数组, mid+1..right 为旋转排序数组
                if (first <= target && target < nums[mid]) {
                    right = mid - 1
                } else {
                    left = mid + 1
                }
            } else {
                // left..mid 为旋转排序数组, mid+1..right 为排序数组
                if (nums[mid] < target && target <= last) {
                    left = mid + 1
                } else {
                    right = mid - 1
                }
            }
        }
        return -1
    }
}

/// 74. 搜索二维矩阵
class Solution74 {
    fun searchMatrix(matrix: Array<IntArray>, target: Int): Boolean {
        val rowCount = matrix.size
        val colCount = matrix[0].size

        var left = 0
        var right = rowCount * colCount - 1

        while (left <= right) {
            val mid = (left + right) ushr 1
            val row = mid / colCount
            val col = mid % colCount
            val midValue = matrix[row][col]
            if (midValue == target) {
                return true
            }
            if (midValue < target) {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
        return false
    }
}

/// 933. 最近的请求次数
class Solution933 {
    class RecentCounter() {
        /**
         * Your RecentCounter object will be instantiated and called as such:
         * var obj = RecentCounter()
         * var param_1 = obj.ping(t)
         */
        val timeIntervals = LinkedList<Int>()
        val capacity = 3001

        fun ping(t: Int): Int {
            val target = t - 3000
            timeIntervals.offerLast(t)
            while (timeIntervals.peekFirst() < target) {
                timeIntervals.removeFirst()
            }
            return timeIntervals.size
        }
    }
}


/// 699. 掉落的方块
class Solution699 {
    fun fallingSquares(positions: Array<IntArray>): List<Int> {
        val n = positions.size
        val heights: MutableList<Int> = ArrayList()
        for (i in 0 until n) {
            var curHeight = positions[i][1]
            val curLeft = positions[i][0]
            val curRight = curLeft + curHeight - 1
            for (j in 0 until i) {
                val preHeight = positions[j][1]
                val preLeft = positions[j][0]
                val preRight = preLeft + preHeight - 1
                if (curRight >= preLeft && preRight >= curLeft) { // 判断重合
                    curHeight = Math.max(
                        curHeight,
                        heights[j] + positions[i][1]
                    ) // positions[i][1]不能用curHeight代替, 因为curHeight会变换.
                }
            }
            heights.add(curHeight)
        }
        for (i in 1 until n) {
            heights[i] = Math.max(heights[i], heights[i - 1])
        }
        return heights
    }

    /// 有序集合
    fun fallingSquares_SortedMap(positions: Array<IntArray>): List<Int> {
        val n = positions.size
        val ret = ArrayList<Int>()
        val heightMap = TreeMap<Int, Int>() // SortedMap 对 key 值进行排序的 Map
        heightMap[0] = 0

        for (i in 0 until n) {
            val size = positions[i][1]
            val left = positions[i][0]
            val right = left + size - 1
            val lp = heightMap.higherKey(left)  // 大于 left 的最小 key 值
            val rp = heightMap.higherKey(right) // 大于 right 的最小 key 值
            val prevRightKey = if (rp != null) heightMap.lowerKey(rp) else heightMap.lastKey()
            // 记录 right + 1 对应的堆叠高度 (如果 right + 1 不在 heightMap 中)
            var rHeight = if (prevRightKey != null) heightMap[prevRightKey]!! else 0

            // 更新第 i 个掉落的访客的堆叠高度
            var height = 0
            val prevLeftKey = if (lp != null) heightMap.lowerKey(lp) else heightMap.lastKey()

            val tail = if (prevLeftKey != null) heightMap.tailMap(prevLeftKey) else heightMap
            for (entry in tail.entries) {
                if (entry.key == rp) break
                height = maxOf(height, entry.value + size)
            }

            // 清除 heightMap 中位于 (left, right] 内的点
            val keySet = TreeSet(tail.keys)
            for (tmp in keySet) {
                if (lp == null || tmp < lp) continue
                if (rp != null && tmp >= rp) break
                heightMap.remove(tmp)
            }

            heightMap[left] = height
            if (rp == null || rp != right + 1) {
                heightMap[right + 1] = rHeight
            }
            ret.add(if (i > 0) maxOf(ret[i - 1], height) else height)
        }
        return ret
    }
}

/// 面试题 17.11. 单词距离
class SolutionMS17_11 {
    fun findClosest(words: Array<String>, word1: String, word2: String): Int {
        var word1Index: Int? = null
        var word2Index: Int? = null
        var min = Int.MAX_VALUE
        for (i in words.indices) {
            val isWord1 = words[i] == word1
            val isWord2 = words[i] == word2
            if (isWord1) {
                word1Index = i
            }
            if (isWord2) {
                word2Index = i
            }
            if ((isWord1 || isWord2) && word1Index != null && word2Index != null) {
                if (word1Index < word2Index) {
                    min = minOf(min, word2Index!! - word1Index!!)
                    word1Index = null
                } else {
                    min = minOf(min, word1Index!! - word2Index!!)
                    word2Index = null
                }
            }
        }
        return min
    }
}

/// 153. 寻找旋转排序数组中的最小值
class Solution153 {
    fun findMin_(nums: IntArray, from: Int = 0, to: Int = nums.size - 1): Int {
        var left = from
        var right = to
        if (left < right) {
            val mid = (left + right) ushr 1
            val midValue = nums[mid]
            if (midValue >= nums[from]) {
                // from..mid 有序, mid+1..to 旋转
                val leftMin = nums[from]
                val rightMin = findMin_(nums, mid + 1, to)
                return minOf(leftMin, rightMin)
            } else { // midValue < nums[from]
                // from..mid 旋转, mid+1..to 有序
                val leftMin = findMin_(nums, from, mid)
                val rightMin = nums[mid + 1]
                return minOf(leftMin, rightMin)
            }
        }
        return nums[left]
    }

    fun findMin(nums: IntArray): Int {
        return findMin_(nums)
    }

    fun findMin_FastBinary(nums: IntArray): Int {
        val first = nums[0]
        val last = nums[nums.size - 1]
        if (first <= last) return first

        // first > last
        var left = 0
        var right = nums.size - 1
        while (left < right) {
            val mid = (left + right) ushr 1
            val midVal = nums[mid]
            if (midVal >= first) {
                // 前半部分 有序, 后半部分 [mid + 1, to] 旋转, 最小值在 后半部分
                left = mid + 1
            } else /* if (midVal < last)*/ { // midVal < first && midVal < last
                // 前半部分 [from, mid] 翻转, 后半部分 有序, 最小值在 前半部分
                right = mid
            } /*else { // midVal < first && midVal > last
                //(1) midVal < first, 说明 [0..mid] 是旋转数组
                //(2) midVal > last, 说明 [mid..nums.size-1] 是旋转数组
                //(1)和(2)是互相冲突的, 所以此处不可能被执行到.
            }*/
        }
        return nums[left]
    }
}

/// 162. 寻找峰值
class Solution162 {
    fun findPeakElement(nums: IntArray): Int {
        var left = 0
        var right = nums.size - 1

        while (left <= right) {
            var mid = (left + right) ushr 1
            val midVal = nums[mid]
            val leftLessThanMid = (mid == 0 || (mid - 1 >= 0 && nums[mid - 1] < midVal))
            val rightLessThanMid =
                (mid == nums.size - 1 || (mid + 1 < nums.size && nums[mid + 1] < midVal))
            if (leftLessThanMid && rightLessThanMid) {
                return mid
            } else if (leftLessThanMid) {
                left = mid + 1
            } else /* if (rightLessThanMid) */ {
                right = mid - 1
            } /* else { // mid 是低谷, 往左或往右都行
                left = mid + 1
                // right = mid - 1
            }*/
        }
        return -1
    }
}

/// 线段树
class SegmentTree(val nums: IntArray) {

    class Node(var start: Int, var end: Int) {
        var data: Int = 0
        var mark = 0

        fun addMark(value: Int) {
            mark += value
        }

        fun clearMark() {
            mark = 0
        }

        override fun toString(): String {
            return "${start} - ${end}"
        }
    }

    var nodes: MutableList<Node?>

    init {
        nodes = MutableList<Node?>(nums.size) { null }
        build(0)
    }

    /// 构建线段树
    fun build(index: Int) {
        var node = nodes[index]
        if (node == null) {
            // 构建根节点
            node = Node(0, nodes.size - 1)
            nodes[0] = node
        }
        assert(node != null)

        if (node.start == node.end) {
            // 如果这个线段你的左断电等于右端点，则这个点是叶子节点
            node.data = nums[node.start]
        } else {
            // 否则递归构造左右子树
            val mid = (node.start + node.end) ushr 1
            val left = (index shl 1) + 1
            val right = (index shl 1) + 2
            nodes[left] = Node(node.start, mid)   // 左子节点线段
            nodes[right] = Node(mid + 1, node.end) // 有子节点线段
            build(left)  // 构造左孩子
            build(right) // 构造右孩子
            node.data = minOf(nodes[left]!!.data, nodes[right]!!.data)
        }
    }

    /// 查询当前区间中待查询区间内的最小值
    /// index: 当前区间下标
    /// start: 待查询区间左端点
    /// end: 待查询区间右端点
    fun query(index: Int, start: Int, end: Int): Int {
        val node = nodes[index]
        if (node == null || node.start > end || node.end < start) return Int.MAX_VALUE // 当前区间和待查询区间没有交集，返回极大值

        if (node.start >= start && node.end <= end) return node.data // 如果当前区间被包含在待查询区间中，直接返回当前区间的最小值，即node.data

        pushDown(index)

        // 递归查询左子树和右子树
        return maxOf(query((index shl 1) + 1, start, end), query((index shl 1) + 2, start, end))
    }

    /// 将更新和标记信息扩展到左右子树
    fun pushDown(index: Int) {
        val node = nodes[index]
        if (node != null && node.mark != 0) {
            val left = (index shl 1) + 1
            val right = (index shl 1) + 2
            nodes[left]!!.addMark(node.mark)  // 更新左子树的标志
            nodes[right]!!.addMark(node.mark) // 更新右子树的标志
            nodes[left]!!.data += node.mark   // 左子树的值加上标志值
            nodes[right]!!.data += node.mark  // 右子树的值加上标志值
            node.clearMark() // 清除当前节点的标志值
        }
    }

    fun update(index: Int, start: Int, end: Int, dataAdd: Int) {
        val node = nodes[index]
        if (node == null || node.start > end || node.end < start) return

        if (node.start >= start && node.end <= end) {
            // 如果当前区间被包含在待查询区间之内， 则当前区间需要更新data, 同时被标记data更新量
            node.data += dataAdd
            node.addMark(dataAdd)
            return
        }

        pushDown(index) // 更新左右子树之前进行扩展操作

        val left = (index shl 1) + 1
        val right = (index shl 1) + 2
        update(left, start, end, dataAdd)
        update(right, start, end, dataAdd)
        node.data = minOf(nodes[left]!!.data, nodes[right]!!.data)
    }
}

/**
 * Example:
 * var li = ListNode(5)
 * var v = li.`val`
 * Definition for singly-linked list.
 * class ListNode(var `val`: Int) {
 *     var next: ListNode? = null
 * }
 */
/// 83. 删除排序链表中的重复元素 II
class Solution83 {
    fun deleteDuplicates(head: ListNode?): ListNode? {
        var prehead = ListNode(-1000)
        prehead.next = head
        var current = prehead
        while (current.next != null) {
            val next = current.next!!
            if (current.`val` == next.`val`) {
                // 删除 next
                current.next = next.next
            } else {
                current = current.next!!
            }
        }
        return prehead.next
    }
}

/// 82. 删除排序链表中的重复元素 II
class Solution82 {
    fun deleteDuplicates(head: ListNode?): ListNode? {
        val prehead = ListNode(-1000)
        prehead.next = head
        var current: ListNode? = prehead
        while (current?.next != null) {
            var next = current!!.next
            while (next != null && next!!.`val` == next!!.next?.`val`) {
                next = next!!.next
            }
            if (current!!.next != next) {
                current!!.next = next?.next
            } else {
                current = current!!.next
            }
        }
        return prehead.next
    }

    fun deleteDuplicates1(head: ListNode?): ListNode? {
        val prehead = ListNode(-1000)
        prehead.next = head
        var current: ListNode? = prehead

        while (current?.next != null) {
            var next = current!!.next
            while (next != null && next!!.`val` == next!!.next?.`val`) next = next!!.next

            if (current!!.next == next) {
                current = next
            } else {
                current!!.next = next?.next
            }
        }

        return prehead.next
    }
}

/// 1021. 删除最外层的括号
class Solution1021 {
    fun removeOuterParentheses(s: String): String {
        val result = StringBuffer()

        var leftCount = 0
        var rightCount = 0
        var start = 0
        for (c in s) {
            if (c == '(') {
                leftCount++
            } else {
                rightCount++
            }
            if (leftCount == rightCount) {
                val end = start + leftCount + rightCount
                result.append(s.substring(start + 1, end - 1))

                leftCount = 0
                rightCount = 0
                start = end
            }
        }

        return result.toString()
    }
}

/// 300. 最长递增子序列
class Solution300 {
    fun lengthOfLIS_dp(nums: IntArray): Int {
        val n = nums.size
        var dp = IntArray(n)
        dp[0] = 1

        var max = dp[0]
        for (i in 1 until n) {
            dp[i] = 1
            for (j in 0 until i) {
                if (nums[j] < nums[i]) {
                    dp[i] = maxOf(dp[i], dp[j] + 1)
                }
            }
            max = maxOf(max, dp[i])
        }

        return max
    }

    // 二分
    fun lengthOfLIS(nums: IntArray): Int {
        val n = nums.size

        var min = IntArray(n + 1) // min[i] 代表长度为i的严格递增子序列最小的最后一位数字
        min[1] = nums[0]
        var len = 1

        for (i in 1 until nums.size) { // 遍历严格递增子序列长度
            if (nums[i] > min[len]) {
                min[++len] = nums[i]
            } else {
                val target = nums[i]
                var l = 1
                var r = len
                var pos = 0
                // 找到第一个小于target的最大值（越大对应的长度越长）
                while (l <= r) {
                    val mid = (l + r) ushr 1
                    if (min[mid] < target && (mid == len || min[mid + 1] >= target)) {
                        pos = mid
                        break
                    }
                    if (min[mid] < target) {
                        l = mid + 1
                    } else { // min[mid] >= target
                        r = mid - 1
                    }
                }
                min[pos + 1] = target
            }
        }
        return len
    }
}

/// 300. 最长递增子序列
class Solution300_20220610 {
    fun lengthOfLIS(nums: IntArray): Int {
        val n = nums.size
        // dp[i] 代表 nums[0..i] 中以 nums[i] 结尾的最长严格递增子序列的长度
        // dp[0] = 1
        // dp[i] = if (nums[i] > nums[i - 1]) dp[i - 1] + 1 else 0
        val dp = IntArray(n)
        dp[0] = 1
        var max = dp[0]
        for (i in 1 until n) {
            dp[i] = 1
            for (j in 0 until i) {
                if (nums[i] > nums[j]) {
                    dp[i] = maxOf(dp[i], dp[j] + 1)
                }
            }
            max = maxOf(max, dp[i])
        }
        return max
    }
}

/// 468. 验证IP地址
class Solution468 {
    fun isValidIPv4Segment(segment: String): Boolean {
        // 空，长度小于1或大于3，无效
        if (segment == null || segment.isEmpty() || segment.length > 3) {
            return false
        }
        // 前导0（长度大于1），无效
        if (segment.length > 1 && segment[0] == '0') return false
        // 包含非0~9字符，无效
        var value = 0
        for (i in segment.indices) {
            val c = segment[i]
            if (c in '0'..'9') {
                value = value * 10 + (c - '0')
            } else {
                return false
            }
        }
        // 数字范围 0~255，有效，否则无效
        return value in 0..255
    }

    fun isValidIPv6Segment(segment: String): Boolean {
        // 空，长度小于1或大于4，无效
        if (segment == null || segment.isEmpty() || segment.length > 4) {
            return false
        }
        // 包含非 0~9，a~f 的字符，无效
        var value = 0
        for (i in segment.indices) {
            val c = segment[i]
            if (c in '0'..'9') {
                value = value * 16 + (c - '0')
            } else if (c in 'a'..'f') {
                value = value * 16 + (c - 'a' + 10)
            } else if (c in 'A'..'F') {
                value = value * 16 + (c - 'A' + 10)
            } else {
                return false
            }
        }
        // 数字范围 0~0xFFFF
        return value in 0..0xFFFF
    }

    fun validIPAddress(queryIP: String): String {
        if (queryIP.contains('.')) {
            // 可能是IPv4
            val segments = queryIP.split(".")
            if (segments.size == 4) {
                var valid = true
                for (segment in segments) {
                    if (!isValidIPv4Segment(segment)) {
                        valid = false
                        break
                    }
                }
                if (valid) {
                    return "IPv4"
                }
            }
        } else if (queryIP.contains(':')) {
            // 可能是IPv6
            val segments = queryIP.split(":")
            if (segments.size == 8) {
                var valid = true
                for (segment in segments) {
                    if (!isValidIPv6Segment(segment)) {
                        valid = false
                        break
                    }
                }
                if (valid) {
                    return "IPv6"
                }
            }
        }
        return "Neither"
    }
}

/// 844. 比较含退格的字符串
class Solution844 {
    fun backspaceCompare1(left: String, right: String): Boolean {
        fun process(s: String): String {
            var sb = StringBuilder()
            for (c in s) {
                if (c == '#') {
                    if (sb.isNotEmpty()) {
                        sb.delete(sb.length - 1, sb.length)
                    }
                } else {
                    sb.append(c)
                }
            }
            return sb.toString()
        }
        return process(left) == process(right)
    }

    fun backspaceCompare(left: String, right: String): Boolean {
        var l = left.length - 1
        var r = right.length - 1
        while (l >= 0 || r >= 0) { // 从后往前遍历
            var hashCount = 0
            while (l >= 0 && left[l] == '#') { // 判断是否可以继续统计 # 号个数和删除左侧非 # 号的字符
                while (l >= 0 && left[l] == '#') { // 记录当前 # 号个数，下标左移
                    hashCount++
                    l--
                }
                while (hashCount > 0 && l >= 0 && left[l] != '#') { // 用 # 号抵消非 # 号字符
                    hashCount--
                    l--
                }
            }
            val lc = if (hashCount > 0 || l < 0) null else left[l] // 取字符

            hashCount = 0
            while (r >= 0 && right[r] == '#') {
                while (r >= 0 && right[r] == '#') {
                    hashCount++
                    r--
                }
                while (hashCount > 0 && r >= 0 && right[r] != '#') {
                    hashCount--
                    r--
                }
            }
            val rc = if (hashCount > 0 || r < 0) null else right[r]

            if (lc != rc) {
                return false
            } else {
                l--
                r--
            }
        }

        return true
    }
}

/// 986. 区间列表的交集
class Solution986 {
    /// 暴力
    fun intervalIntersection_Force(
        firstList: Array<IntArray>,
        secondList: Array<IntArray>
    ): Array<IntArray> {
        var result = mutableListOf<IntArray>()

        for (second in secondList) {
            for (first in firstList) {
                if (first[0] <= second[1] && first[1] >= second[0]) {
                    result.add(intArrayOf(maxOf(first[0], second[0]), minOf(first[1], second[1])))
                }
                if (first[0] > second[1]) {
                    break
                }
            }
        }

        return result.toTypedArray()
    }

    // 双指针
    fun intervalIntersection_TwoPointers(
        firstList: Array<IntArray>,
        secondList: Array<IntArray>
    ): Array<IntArray> {
        var result = mutableListOf<IntArray>()

        var i = 0
        var j = 0
        while (j < secondList.size) {
            while (i < firstList.size) {
                val first = firstList[i]
                val second = secondList[j]
                if (first[0] <= second[1] && first[1] >= second[0]) {
                    result.add(intArrayOf(maxOf(first[0], second[0]), minOf(first[1], second[1])))
                }
                if (first[1] >= second[1]) {
                    break
                }
                i++
            }
            j++
        }

        return result.toTypedArray()
    }
}

/// LCP 55. 采集果实
class SolutionLCP55 {
    fun getMinimumTime(time: IntArray, fruits: Array<IntArray>, limit: Int): Int {
        var totalTime = 0

        for (fruit in fruits) {
            val type = fruit[0]
            val count = fruit[1]
            totalTime += Math.ceil(count * 1.0 / limit).toInt() * time[type]
        }

        return totalTime
    }
}

/// 2279. 装满石头的背包的最大数量
class Solution2279 {
    fun maximumBags(capacity: IntArray, rocks: IntArray, additionalRocks: Int): Int {
        var remains = capacity.mapIndexed { i, c -> c - rocks[i] }.toMutableList()
        remains.sort()
        var fullBagCount = 0
        var rocksCount = additionalRocks
        for (r in remains) {
            if (r <= rocksCount) {
                rocksCount -= r
                fullBagCount++
            }
        }
        return fullBagCount
    }
}

class SolutionBag01 {
    var maxW = Int.MIN_VALUE
    val weight = intArrayOf(2, 2, 4, 6, 3)
    val n = weight.size
    val w = 9
    var mem = Array(n) { BooleanArray(9 + 1) }
    fun recall(i: Int, cw: Int) {
        if (cw == w || i == n) { // cw==w表示装满了，i==n表示物品都考察完了
            if (cw > maxW) maxW = cw
            return
        }

        if (mem[i][cw]) return

        mem[i][cw] = true

        recall(i + 1, cw) // 选择不装入第i个物品
        if (cw + weight[i] <= w) {
            recall(i + 1, cw + weight[i]) // 选择装入第i个物品
        }
    }

    /// 求最大重量
    fun knapsack01(weight: IntArray, w: Int): Int {
        // states[i][j] 代表从第0~i个物品中选出若干物品装入背包，且要保证背包重量为j，这种情况是否可行，
        // 可行则 states[i][j] = true, 不可行则 states[i][j] = false
        // 显然，state[0][0] = true, 如果 weight[0] <= w, 则 state[0][weight[0]]=true
        val states = Array(weight.size) { BooleanArray(w + 1) }
        states[0][0] = true
        if (weight[0] <= w) {
            states[0][weight[0]] = true
        }
        for (i in 1 until weight.size) {
            for (j in 0 until w + 1) {
                // 不把第i个物品放入背包
                if (states[i - 1][j]) {
                    states[i][j] = true
                }
            }
            for (j in 0..(w - weight[i])) {
                // 把第i个物品房屋背包
                if (states[i - 1][j]) {
                    states[i][j + weight[i]] = true
                }
            }
        }
        for (j in w downTo 0) {
            if (states[n - 1][j]) return j
        }
        return 0
    }

    /// 求最大重量
    fun knapsack01_op(weight: IntArray, w: Int): Int {
        // states[j] 代表从第所有物品中选出若干物品装入背包，且要保证背包重量为j，这种情况是否可行，
        // 可行则 states[j] = true, 不可行则 states[j] = false
        // 显然，state[0] = true, 如果 weight[0] <= w, 则 state[weight[0]]=true
        //
        // 按照 knapsack01 的思路，可以从第一个物品开始处理，逐步扩大使用到的物品的个数，实际的思路和 knapsack01 是
        // 一致的，只是进行了空间优化。
        //
        val states = BooleanArray(w + 1)
        states[0] = true
        if (weight[0] <= w) {
            states[weight[0]] = true
        }
        for (i in 1 until weight.size) {
            for (j in (w - weight[i]) downTo 0) {
                // 把第i个物品房屋背包
                if (states[j]) {
                    states[j + weight[i]] = true
                }
            }
        }
        for (j in w downTo 0) {
            if (states[j]) return j
        }
        return 0
    }

    /// 求最大价值
    fun knapsack01Plus(weight: IntArray, value: IntArray, w: Int): Int {
        val n = weight.size
        val states = Array(n) { IntArray(w + 1) { -1 } }
        states[0][0] = 0
        if (weight[0] < w) {
            states[0][weight[0]] = value[0]
        }

        for (i in 1 until n) {
            for (j in 0..w) {
                if (states[i - 1][j] >= 0) states[i][j] = states[i - 1][j]
            }
            for (j in 0..(w - weight[i])) {
                if (states[i - 1][j] >= 0) {
                    val v = states[i - 1][j] + value[i]
                    if (v > states[i - 1][j + weight[i]]) states[i - 1][j + weight[i]] = v
                }
            }
        }

        var maxValue = -1
        for (j in 0 until w + 1) {
            if (states[n - 1][j] > maxValue) maxValue = states[n - 1][j]
        }
        return maxValue
    }

    /// 求最大价值
    fun knapsack01Plus_op(weight: IntArray, value: IntArray, w: Int): Int {
        val n = weight.size
        val states = IntArray(w + 1) { -1 }
        states[0] = 0
        if (weight[0] < w) {
            states[weight[0]] = value[0]
        }

        for (i in 1 until n) {
            for (j in (w - weight[i]) downTo 0) {
                if (states[j] >= 0) {
                    val v = states[j] + value[i]
                    if (v > states[j + weight[i]]) states[j + weight[i]] = v
                }
            }
        }

        var maxValue = -1
        for (j in 0 until w + 1) {
            if (states[j] > maxValue) maxValue = states[j]
        }
        return maxValue
    }
}

/// 416. 分割等和子集
class Solution416 {
    fun canPartition(nums: IntArray): Boolean {
        /// 先计算总和 sum，然后问题转化为从 nums 中选出若干个数，使它们的和为 sum/2
        var sum = nums.sum()
        if (sum and 1 == 1) return false // 奇数，必然无法等分为2份
        sum = sum ushr 1 // 确定目标值

        val n = nums.size
        // states[i][j] 代表从 nums[0..i] 中选出若干数字，且保证它们的和为 j 是否可能。
        val states = Array(n) { BooleanArray(sum + 1) }
        states[0][0] = true
        if (nums[0] <= sum) {
            states[0][nums[0]] = true
        }
        for (i in 1 until n) { // 一次迭代决定是否需要选择一个数，第i次即nums[i]
            for (j in 0 until sum + 1) {
                if (states[i - 1][j]) states[i][j] = true // 不选择第i个数
            }
            for (j in (sum - nums[i]) downTo 0) {
                if (states[i - 1][j]) states[i][j + nums[i]] = true
            }
        }
        return states[n - 1][sum]
    }

    fun canPartition_op(nums: IntArray): Boolean {
        /// 先计算总和 sum，然后问题转化为从 nums 中选出若干个数，使它们的和为 sum/2
        var sum = nums.sum()
        if (sum and 1 == 1) return false // 奇数，必然无法等分为2份
        sum = sum ushr 1 // 确定目标值

        val n = nums.size
        // states[i][j] 代表从 nums[0..i] 中选出若干数字，且保证它们的和为 j 是否可能。
        val states = BooleanArray(sum + 1)
        states[0] = true
        if (nums[0] <= sum) {
            states[nums[0]] = true
        }
        for (i in 1 until n) { // 一次迭代决定是否需要选择一个数，第i次即nums[i]
            for (j in (sum - nums[i]) downTo 0) {
                if (states[j]) states[j + nums[i]] = true
            }
        }
        return states[sum]
    }
}

/// 1049. 最后一块石头的重量 II
class Solution1049 {
    fun lastStoneWeightII(stones: IntArray): Int {
        val summ = stones.sum()
        var sum = summ / 2
        val n = stones.size

        // states[i][j] 代表从stones[0..i] 中选出若干石头，满足它们的重量和为j，states[i][j]记录此时的重量和。
        val states = Array(n) { IntArray(sum + 1) { -1 } }
        states[0][0] = 0
        if (stones[0] <= sum) {
            states[0][stones[0]] = stones[0]
        }
        for (i in 1 until n) {
            // 不选择第 i 个石头
            for (j in 0 until sum + 1) {
                if (states[i - 1][j] >= 0) states[i][j] = states[i - 1][j]
            }

            // 选择第 i 个石头，则目标重量为 0..sum-stones[i]
            for (j in sum - stones[i] downTo 0) {
                if (states[i - 1][j] >= 0) {
                    val v = states[i - 1][j] + stones[i]
                    if (v > states[i][j + stones[i]]) {
                        states[i][j + stones[i]] = v
                    }
                }
            }
        }
        var maxValue = 0
        for (j in sum downTo 0) {
            if (states[n - 1][j] > maxValue) maxValue = states[n - 1][j]
        }
        return summ - maxValue * 2
    }

    fun lastStoneWeightII_op(stones: IntArray): Int {
        val summ = stones.sum()
        var sum = summ / 2
        val n = stones.size

        // states[i][j] 代表从stones[0..i] 中选出若干石头，满足它们的重量和为j，states[i][j]记录此时的重量和。
        val states = IntArray(sum + 1) { -1 }
        states[0] = 0
        if (stones[0] <= sum) {
            states[stones[0]] = stones[0]
        }
        for (i in 1 until n) {
            // 选择第 i 个石头，则目标重量为 0..sum-stones[i]
            for (j in sum - stones[i] downTo 0) {
                if (states[j] >= 0) {
                    val v = states[j] + stones[i]
                    if (v > states[j + stones[i]]) {
                        states[j + stones[i]] = v
                    }
                }
            }
        }
        var maxValue = 0
        for (j in sum downTo 0) {
            if (states[j] > maxValue) maxValue = states[j]
        }
        return summ - maxValue * 2
    }
}

/**
 * Example:
 * var ti = TreeNode(5)
 * var v = ti.`val`
 * Definition for a binary tree node.
 * class TreeNode(var `val`: Int) {
 *     var left: TreeNode? = null
 *     var right: TreeNode? = null
 * }
 */
/// 1022. 从根到叶的二进制数之和
class Solution1022 {
    var numbers = mutableListOf<Int>()
    fun travel(cur: Int, node: TreeNode?) {
        if (node == null) {
            return
        }

        var cur = (cur shl 1) + node!!.`val`
        if (node!!.left == null && node!!.right == null) {
            numbers.add(cur)
            return
        }

        if (node!!.left != null) {
            travel(cur, node!!.left)
        }
        if (node!!.right != null) {
            travel(cur, node!!.right)
        }
    }

    fun sumRootToLeaf(root: TreeNode?): Int {
        travel(0, root)
        return numbers.sum()
    }
}

/// 438. 找到字符串中所有字母异位词
class Solution438 {
    fun findAnagrams(s: String, p: String): List<Int> {
        if (p.length > s.length) return listOf()

        val currentChars = IntArray(26)
        val chars = IntArray(26)
        fun isEqual(): Boolean {
            for (i in currentChars.indices) {
                if (chars[i] != currentChars[i]) return false
            }
            return true
        }

        for (c in p) {
            chars[c - 'a'] += 1
        }

        val results = mutableListOf<Int>()
        for (i in 0 until p.length - 1) {
            currentChars[s[i] - 'a'] += 1
        }
        currentChars[s[p.length - 1] - 'a'] += 1
        if (isEqual()) {
            results.add(0)
        }
        for (i in p.length until s.length) {
            currentChars[s[i - p.length] - 'a'] -= 1
            currentChars[s[i] - 'a'] += 1
            if (isEqual()) {
                results.add(i - p.length + 1)
            }
        }
        return results
    }
}

/// 713. 乘积小于 K 的子数组
class Solution713 {
    fun numSubarrayProductLessThanK(nums: IntArray, k: Int): Int {
        var count = 0
        for (i in 0 until nums.size) {
            var product = nums[i]
            var j = i + 1
            while (product < k) {
                count++
                if (j >= nums.size) break
                product *= nums[j++]
            }
        }
        return count
    }

    fun numSubarrayProductLessThanK_SlideWin(nums: IntArray, k: Int): Int {
        val n = nums.size
        var count = 0
        var product = 1
        var l = 0
        for (r in 0 until n) {
            product *= nums[r]
            while (product >= k && l <= r) {
                product /= nums[l]
                l++
            }
            count += r - l + 1 // 字符串 abcd 以 d 结尾的子串个数为字符串 abcd 的长度
        }
        return count
    }
}

/// 209. 长度最小的子数组
class Solution209 {
    fun minSubArrayLen(target: Int, nums: IntArray): Int {
        // 以nums中第i个数结尾的子数组，其中子数组和大于等于target长度最小的，记录下长度 （总是取最小）
        var length = Int.MAX_VALUE

        var sum = 0
        var left = 0
        for (right in 0 until nums.size) {
            sum += nums[right]
            while (sum >= target && left <= right) {
                length = minOf(length, right - left + 1)
                sum -= nums[left]
                left++
            }
        }

        return if (length == Int.MAX_VALUE) 0 else length
    }
}

/// 525. 连续数组
class Solution525 {
    fun findMaxLength(nums: IntArray): Int {
        // 以 nums[i] 结尾的子数组中，含有相同0和1的，且长度最长的，获取其长度

        val n = nums.size

        // sum[i] 代表 nums[0..i] 特殊统计和（特殊统计和：将0视为-1，1不变）
        val sum = IntArray(n)
        sum[0] = if (nums[0] == 0) -1 else 1
        for (i in 1 until n) {
            sum[i] = sum[i - 1] + (if (nums[i] == 0) -1 else 1)
        }

        for (len in (if (n and 1 == 1) n - 1 else n) downTo 1 step 2) {
            for (i in 0..(n - len)) {
                val b = sum.getOrElse(i - 1) { 0 }
                val a = sum[i + len - 1]
                if (a - b == 0) {
                    return len
                }
            }
        }

        return 0
    }

    /// 前缀和+哈希表
    fun findMaxLength_preSum(nums: IntArray): Int {
        val counterIndices = mutableMapOf<Int, Int>()
        var counter = 0
        counterIndices[counter] = -1 // 下标0坐标那个位置
        var maxLen = 0

        for (i in nums.indices) {
            if (nums[i] == 1) {
                counter++
            } else {
                counter--
            }
            if (counterIndices.containsKey(counter)) {
                val prevIndex = counterIndices[counter]!!
                maxLen = maxOf(maxLen, i - prevIndex)
            } else {
                counterIndices[counter] = i
            }
        }

        return maxLen
    }
}

/// 325. 和等于 k 的最长子数组长度
class Solution325 {
    fun maxSubArrayLen(nums: IntArray, k: Int): Int {
        val n = nums.size
        /// 记录前缀和，遍历到 nums[i]，则计算 nums[0..i] 和为 sumBig,
        /// 只需要在之前已经记录的前缀和中找到值为 sumBig - k 的，则说明可以构成目标和为 k 的连续子数组。
        val sumIndices = mutableMapOf<Int, Int>()
        sumIndices[0] = -1
        var maxLen = 0
        var sum = 0

        for (i in 0 until n) {
            sum += nums[i]

            if (sumIndices.containsKey(sum - k)) {
                val prevIndex = sumIndices[sum - k]!!
                maxLen = maxOf(maxLen, i - prevIndex)
            }

            if (!sumIndices.containsKey(sum)) {
                sumIndices[sum] = i
            }
        }
        return maxLen
    }
}

/// 547. 省份数量
class Solution547 {
    fun findCircleNum(isConnected: Array<IntArray>): Int {
        val rowCount = isConnected.size
        val colCount = isConnected[0].size

        fun wipe(row: Int): Int {
            var count = 0
            if (isConnected[row][row] == 1) {
                isConnected[row][row] = 0
                count = 1
            }
            for (col in 0 until colCount) {
                if (isConnected[row][col] == 1) {
                    isConnected[row][col] = 0
                    isConnected[col][row] = 0
                    wipe(col)
                }
            }
            return count
        }

        var count = 0
        for (row in 0 until rowCount) {
            if (isConnected[row][row] == 1)
                count += wipe(row)
        }

        return count
    }
}

/// 269. 火星词典
class Solution269 {
    /// 提取所有字符 => 搜索字符与字符之前是否存在前后关系（构造有向图矩阵）=> 拓扑排序（找到环则说明无解）=> 输出拓扑排序结果字符串
    fun alienOrder(words: Array<String>): String {
        // wordRelation[i][j] 代表字母 'a'+i 和字母 'a'+j 的关系是否是小于关系，即是否 'a'+i < 'a'+j
        // 默认 wordRelations[i][i] = false
        var wordRelation = Array(26) { BooleanArray(26) }
        var charSet = mutableSetOf<Char>()

        val sortedWordsList = LinkedList(listOf(words))
        while (sortedWordsList.isNotEmpty()) {
            val sortedWords = sortedWordsList.pollFirst()
            if (sortedWords.isEmpty()) continue

            var suffixWords = mutableListOf<String>()

            var hasNonempty = false
            var prev: Char? = null
            for (word in sortedWords) {
                if (word.isEmpty()) {
                    if (hasNonempty) {
                        // ""(空串)排在非空串后面，比非空串大，不符合排序要求
                        return ""
                    }
                    continue
                }
                hasNonempty = true
                val c = word[0]
                charSet.add(c)
                if (prev == null) {
                    prev = c
                    suffixWords.add(word.substring(1))
                } else if (prev != c) {
                    wordRelation[prev - 'a'][c - 'a'] = true
                    prev = c

                    if (suffixWords.size >= 1) {
                        sortedWordsList.offerLast(suffixWords.toTypedArray())
                    }

                    suffixWords.clear()
                    suffixWords.add(word.substring(1))
                } else {
                    suffixWords.add(word.substring(1))
                }
            }
            if (suffixWords.size >= 1) {
                sortedWordsList.offerLast(suffixWords.toTypedArray())
            }
        }

        val chars = mutableListOf<Char>()
        while (charSet.isNotEmpty()) {
            var lookFor: Char? = null
            for (ci in charSet) {
                var inDegree = 0
                for (cj in charSet) {
                    if (wordRelation[cj - 'a'][ci - 'a']) {
                        inDegree++
                    }
                }
                if (inDegree == 0) {
                    lookFor = ci
                    break
                }
            }

            if (lookFor != null) {
                charSet.remove(lookFor)
                // 找到一个前驱节点 i
                chars.add(lookFor)
            } else {
                return "" // 存在环
            }
        }

        return chars.joinToString("")
    }
}

/// 473. 火柴拼正方形
class Solution473 {
    /// 内存超过限制
    fun makesquare__(matchsticks: IntArray): Boolean {
        if (matchsticks.size < 4) return false

        var sum = matchsticks.sum()
        val target = sum / 4
        if (sum != target * 4) return false

        // 从 matchsticks 中选出若干个，使得它们的和为 target
        fun match(matchsticks: IntArray, target: Int): List<Int> {
            val n = matchsticks.size
            val NOT_CHOOSE: Byte = 1
            val CHOOSE: Byte = 2
            // states[i][j] = 1/2 代表从 matchsticks[0..i] 中选出若干个，它们的长度和等于 j，其中1代表不选择第i个火柴，2代表选择第i个火柴；否则 state[i][j] = 0
            val states = Array(n) { ByteArray(target + 1) }
            states[0][0] = NOT_CHOOSE
            if (matchsticks[0] <= target) {
                states[0][matchsticks[0]] = CHOOSE
            }

            for (i in 1 until n) {
                // 不选第i个火柴
                for (j in 0 until target) { // 遍历火柴长度，从 0 到目标值 target
                    if (states[i - 1][j] > 0) states[i][j] = NOT_CHOOSE
                }
                // 选第i个火柴
                for (j in target - matchsticks[i] downTo 0) {
                    if (states[i - 1][j] > 0) {
                        states[i][j + matchsticks[i]] = CHOOSE
                    }
                }
            }

            val results = mutableListOf<Int>()
            for (ii in n - 1 downTo 0) {
                if (states[ii][target] > 0) {
                    // 从 matchsticks 中选出若干个火柴可以构造长度为 target 的边，关键问题是怎么知道选了那几个？
                    var indexState = 0
                    var t = target
                    for (i in ii downTo 0) {
                        if (states[i][t] == CHOOSE) {
                            indexState = indexState or (1 shl i)
                            t -= matchsticks[i]
                        }
                    }
                    results.add(indexState)
                }
            }

            return results
        }

        var selectedList1 = match(matchsticks, target)
        for (selected1 in selectedList1) {
            var sticks2: IntArray? =
                matchsticks.filterIndexed { i, v -> (1 shl i) and selected1 == 0 }
                    .toIntArray() // 取出没有被选中的
            if (sticks2!!.size < 3) continue

            val selectedList2 = match(sticks2!!, target)
            sticks2 = null
            if (selectedList2.isEmpty()) continue

            for (selected2 in selectedList2) {
                var sticks3: IntArray? =
                    matchsticks.filterIndexed { i, v -> (1 shl i) and (selected1 and selected2) == 0 }
                        .toIntArray()
                if (sticks3!!.size < 2) continue

                val selectedList3 = match(sticks3!!, target)
                sticks3 = null
                if (selectedList3.isEmpty()) continue

                return true
            }
        }
        return false
    }

    /// 回溯
    fun makesquare_Recall(matchsticks: IntArray): Boolean {
        if (matchsticks.size < 4) return false

        var sum = matchsticks.sum()
        val target = sum / 4

        if (sum != target * 4) return false

        matchsticks.sort()

        val edges = intArrayOf(0, 0, 0, 0)
        fun dfs(index: Int): Boolean {
            if (index == -1) return true // index == matchsticks.size 无对应的火柴，无需操作，所以直接返回true
            for (i in edges.indices) { // 时间复杂度 O(4^n)
                edges[i] += matchsticks[index]
                if (edges[i] <= target && dfs(index - 1)) { // 深度优先搜索，为了减少深度搜索，可以让 edges[i] <= target 尽快为 false => 可以想到的是让 matchsticks 排序，然后从后往前去搜索。
                    return true
                }
                edges[i] -= matchsticks[index] // 回溯
            }
            return false // 搜索完成仍没发现可能的解，返回 false
        }

        return dfs(matchsticks.size - 1)
    }

    /// 状态压缩+动态规划
    fun makesquare_DP(matchsticks: IntArray): Boolean {
        if (matchsticks.size < 4) return false
        val sum = matchsticks.sum()
        if (sum % 4 != 0) return false

        val target = sum / 4
        val n = matchsticks.size
        val stateCount = 1 shl n // 总共有 2^n 个状态
        // dp[s]默认为-1，代表状态s不符合要求，一开始只有 dp[0] 符合要求（初始设定）
        // dp[s]代表状态s下，已选火柴长度 % target，所求结果为 dp[stateCount - 1] == 0 (代表选上了所有火柴后)
        // 选择火柴 i 时会保证 dp[s1] + matchstick[i] <= target，否则不选火柴 i
        val dp = IntArray(stateCount) { -1 }
        dp[0] = 0 // 所有火柴都不选，长度为0
        for (s1 in 1 until stateCount) {
            for (i in 0 until n) {
                if ((1 shl i) and s1 == 0) continue
                // 去除状态 s 中的第 i 个二进制位（从低到高），得到状态 s0
                val s0 = s1 and (1 shl i).inv()
                if (dp[s0] >= 0 && dp[s0] + matchsticks[i] <= target) {
                    dp[s1] = (dp[s0] + matchsticks[i]) % target
                    break // ???只需要找打一个s0即可
                }
            }
        }
        return dp[stateCount - 1] == 0
    }
}

/// 572. 另一棵树的子树
class Solution572 {
    fun isEqual(n1: TreeNode?, n2: TreeNode?): Boolean {
        if (n1 == n2) return true
        if (n1 == null || n2 == null) return false
        return n1!!.`val` == n2!!.`val` && isEqual(n1!!.left, n2!!.left) && isEqual(
            n1!!.right,
            n2!!.right
        )
    }

    fun isSubtree(root: TreeNode?, subRoot: TreeNode?): Boolean {
        return isEqual(root, subRoot) || (root?.left != null && isSubtree(
            root!!.left,
            subRoot
        )) || (root?.right != null && isSubtree(root!!.right, subRoot))
    }
}

/// 117. 填充每个节点的下一个右侧节点指针 II
class Solution117 {
    class Node(var `val`: Int) {
        var left: Node? = null
        var right: Node? = null
        var next: Node? = null
    }

    fun conn(root: Node?) {
        if (root == null || (root.left == null && root.right == null)) return

        if (root.right != null) {
            val right = root.right!!
            var next = root.next
            while (right.next == null && next != null) {
                right.next = next!!.left ?: next!!.right
                next = next!!.next
            }
        }

        if (root.left != null) {
            val left = root.left!!
            left.next = root.right
            var next = root.next
            while (left.next == null && next != null) {
                left.next = next!!.left ?: next!!.right
                next = next!!.next
            }
        }

        conn(root!!.right)
        conn(root!!.left)
    }

    fun connect(root: Node?): Node? {
        conn(root)
        return root
    }

    fun connect_UseNextPointer(root: Node?): Node? {
        var theroot = root
        var lastUpdated = false
        var last: Node? = null

        var root = root
        while (root != null) {
            if (root.left != null) {
                if (!lastUpdated) {
                    last = root.left
                    lastUpdated = true
                }

                val left = root.left!!
                left.next = root.right
                var next = root.next
                while (left.next == null && next != null) {
                    left.next = next!!.left ?: next!!.right
                    next = next!!.next
                }
            }

            if (root.right != null) {
                if (!lastUpdated) {
                    last = root.right
                    lastUpdated = true
                }

                val right = root.right!!
                var next = root.next
                while (right.next == null && next != null) {
                    right.next = next!!.left ?: next!!.right
                    next = next!!.next
                }
            }

            root = root.next
            if (root == null) {
                lastUpdated = false
                root = last
                last = null
            }
        }

        return theroot
    }
}

/// 338. 比特位计数
class Solution338 {
    // DP
    fun countBits(n: Int): IntArray {
        val results = IntArray(n + 1)
        var highBits = 0
        for (i in 1..n) {
            if (i and (i - 1) == 0) {
                highBits = i
            }
            results[i] = results[i - highBits] + 1
        }
        return results
    }

    fun countBits_Shift(n: Int): IntArray {
        val results = IntArray(n + 1)
        for (i in 1..n) {
            results[i] = results[i ushr 1] + (i and 1)
        }
        return results
    }

    fun countBits_NandN_1(n: Int): IntArray {
        val results = IntArray(n + 1)
        for (i in 1..n) {
            results[i] = results[i and (i - 1)] + 1
        }
        return results
    }
}

/// 450. 删除二叉搜索树中的节点
class Solution450 {
    class TreeNode(var `val`: Int) {
        var left: TreeNode? = null
        var right: TreeNode? = null
    }

    fun search(root: TreeNode?, key: Int): Pair<TreeNode?, TreeNode?> {
        if (root == null) return Pair(null, null)

        var current = root
        if (current.`val` == key) {
            return Pair(null, current)
        }

        while (current != null) {
            if (current!!.left?.`val` == key) {
                return Pair(current, current!!.left)
            }
            if (current!!.right?.`val` == key) {
                return Pair(current, current!!.right)
            }
            if (current.`val` < key) {
                current = current!!.right
            } else {
                current = current!!.left
            }
        }

        return Pair(null, null)
    }

    fun findMin(root: TreeNode?): TreeNode? {
        if (root == null) return null
        var current = root!!
        while (current.left != null) {
            current = current.left!!
        }
        return current
    }

    fun deleteNode(root: TreeNode?, key: Int): TreeNode? {
        var theroot = root

        // 找到目标节点和它的父节点
        val (father, node) = search(root, key)
        // 没有找到目标节点，直接不处理
        if (node == null) return theroot

        // node != null，是根节点
        if (father == null) {
            theroot = node.right
            if (theroot == null) {
                theroot = node.left
            } else {
                findMin(node.right)!!.left = node.left
            }
            return theroot
        }
        // node != null，非根节点
        // node 是 father 的左节点
        if (father.left == node) {
            father.left = node.right
            if (father.left == null) {
                father.left = node.left
            } else {
                findMin(node.right)!!.left = node.left
            }
            return theroot
        }
        // node 是 father 的右节点
        // father.right == node
        father.right = node.right
        if (father.right == null) {
            father.right = node.left
        } else {
            findMin(node.right)!!.left = node.left
        }

        return theroot
    }
}

/// 1091. 二进制矩阵中的最短路径
class Solution {
    fun shortestPathBinaryMatrix(grid: Array<IntArray>): Int {
        val rowCount = grid.size
        val colCount = grid[0].size

        if (grid[0][0] == 1 || grid[rowCount - 1][colCount - 1] == 1) return -1

        val directions = listOf(
            Pair(1, 1),
            Pair(1, 0),
            Pair(0, 1),
            Pair(1, -1),
            Pair(-1, 1),
            Pair(-1, 0),
            Pair(0, -1),
            Pair(-1, -1)
        )

        val distance = Array(rowCount) { IntArray(colCount) { Int.MAX_VALUE } }
        val willSearch = LinkedList<Pair<Int, Int>>()


        distance[0][0] = 1 // 从 (0,0) 到 (0,0) 经过了它自己，即1个节点
        willSearch.offer(Pair(0, 0))

        theWhile@ while (willSearch.isNotEmpty()) {
            val (row, col) = willSearch.poll()

            for ((dr, dc) in directions) {
                val r = row + dr
                val c = col + dc
                if (r < 0 || r >= rowCount || c < 0 || c >= colCount || grid[r][c] == 1 || distance[r][c] != Int.MAX_VALUE) continue
                distance[r][c] = minOf(distance[r][c], distance[row][col] + 1)

                if (r == rowCount - 1 && c == colCount - 1) {
                    break@theWhile
                }

                willSearch.offer(Pair(r, c))
            }
        }

        if (distance[rowCount - 1][colCount - 1] == Int.MAX_VALUE) return -1
        return distance[rowCount - 1][colCount - 1]
    }
}

/// 130. 被围绕的区域
class Solution130 {
    fun solve(board: Array<CharArray>): Unit {
        val rowCount = board.size
        val colCount = board[0].size

        val excludeO = mutableSetOf<Pair<Int, Int>>()
        val queue = LinkedList<Pair<Int, Int>>()
        val visited = Array(rowCount) { BooleanArray(colCount) }

        for (row in setOf(0, rowCount - 1)) {
            for (col in 0 until colCount) {
                visited[row][col] = true
                if (board[row][col] == 'O') {
                    val p = Pair(row, col)
                    excludeO.add(p)
                    queue.offer(p)
                }
            }
        }
        for (col in setOf(0, colCount - 1)) {
            for (row in 1 until rowCount - 1) {
                visited[row][col] = true
                if (board[row][col] == 'O') {
                    val p = Pair(row, col)
                    excludeO.add(p)
                    queue.offer(p)
                }
            }
        }

        val directions = listOf(Pair(1, 0), Pair(-1, 0), Pair(0, 1), Pair(0, -1))
        while (queue.isNotEmpty()) {
            val (row, col) = queue.poll()
            for ((dr, dc) in directions) {
                val r = row + dr
                val c = col + dc
                if (r < 0 || r >= rowCount || c < 0 || c >= colCount || visited[r][c]) continue

                visited[r][c] = true

                if (board[r][c] == 'X') continue

                val pp = Pair(r, c)
                excludeO.add(pp)
                queue.offer(pp)
            }
        }

        for (row in 0 until rowCount) {
            for (col in 0 until colCount) {
                if (board[row][col] == 'O' && !excludeO.contains(Pair(row, col))) {
                    board[row][col] = 'X'
                }
            }
        }
    }

    fun solve_dfs(board: Array<CharArray>): Unit {
        val rowCount = board.size
        val colCount = board[0].size

        fun dfs(row: Int, col: Int) {
            if (row < 0 || row >= rowCount || col < 0 || col >= colCount || board[row][col] != 'O') return

            board[row][col] = 'M'
            dfs(row + 1, col)
            dfs(row - 1, col)
            dfs(row, col + 1)
            dfs(row, col - 1)
        }

        for (col in 0 until colCount) {
            dfs(0, col)
            dfs(rowCount - 1, col)
        }
        for (row in 1 until rowCount - 1) {
            dfs(row, 0)
            dfs(row, colCount - 1)
        }
        for (row in 0 until rowCount) {
            for (col in 0 until colCount) {
                if (board[row][col] == 'M') {
                    board[row][col] = 'O'
                } else if (board[row][col] == 'O') {
                    board[row][col] = 'X'
                }
            }
        }
    }
}

/// 797. 所有可能的路径
class Solution797 {
    fun allPathsSourceTarget(graph: Array<IntArray>): List<List<Int>> {
        val n = graph.size
        var results = mutableListOf<List<Int>>()

        fun dfs(i: Int, current: MutableList<Int>) {
            if (i == n - 1) {
                results.add(current.toList())
                return
            }

            for (j in graph[i]) {
                current.add(j)
                dfs(j, current)
                current.removeAt(current.size - 1)
            }
        }

        dfs(0, mutableListOf<Int>(0))
        return results
    }
}

/// 85. 最大矩形
class Solution85 {
    /// 暴力遍历
    fun maximalRectangle_V(matrix: Array<CharArray>): Int {
        val rowCount = matrix.size
        val colCount = matrix[0].size

        /// 构造left矩阵，left[i][j]代表点(i, j)左侧连续1的个数（包括它自己的1），如果matrix[i][j]=0，则left[i][j]=0
        val left = Array(rowCount) { IntArray(colCount) }
        for (r in 0 until rowCount) {
            for (c in 0 until colCount) {
                if (c == 0) {
                    left[r][c] = matrix[r][c] - '0'
                } else if (matrix[r][c] == '1') {
                    left[r][c] = left[r][c - 1] + 1
                }
            }
        }

        var maxArea = 0
        for (r in 0 until rowCount) {
            for (c in 0 until colCount) {
                if (matrix[r][c] == '0') continue

                var width = left[r][c]
                for (k in r downTo 0) {
                    if (matrix[k][c] == '0') break
                    width = minOf(width, left[k][c])
                    maxArea = maxOf(maxArea, width * (r - k + 1))
                }

            }
        }
        return maxArea
    }

    /// 单调栈
    fun maximalRectangle(matrix: Array<CharArray>): Int {
        val rowCount = matrix.size
        val colCount = matrix[0].size
        var maxArea = 0

        val heights = IntArray(colCount + 2)
        for (r in 0 until rowCount) {
            for (c in 0 until colCount) {
                if (matrix[r][c] == '1') {
                    heights[c + 1] += 1
                } else {
                    heights[c + 1] = 0
                }
            }
            maxArea = maxOf(maxArea, largestRectangleArea(heights))
        }

        return maxArea
    }

    fun largestRectangleArea(heights: IntArray): Int {
        var maxArea = 0
        val stack = Stack<Int>()
        for (i in heights.indices) {
            if (stack.isEmpty() || heights[stack.peek()] <= heights[i]) {
                stack.push(i)
            } else {
                var top = -1
                while (stack.isNotEmpty() && heights[stack.peek()] > heights[i]) {
                    top = stack.pop()
                    maxArea = maxOf(maxArea, heights[top] * (i - 1 - stack[stack.size - 1]))
                }
                stack.push(i)
            }
        }

        return maxArea
    }
}

/// 84. 柱状图中最大的矩形
class Solution84 {
    fun largestRectangleArea(heights: IntArray): Int {
        var maxArea = 0
        val stack = LinkedList<Int>()
        for (i in heights.indices) {
            if (stack.isEmpty() || heights[stack.peekLast()] < heights[i]) {
                stack.offerLast(i)
            } else {
                var top = -1
                while (stack.isNotEmpty() && heights[stack.peekLast()] >= heights[i]) {
                    top = stack.pollLast()
                    maxArea = maxOf(maxArea, heights[top] * (i - top))
                }
                heights[top] = heights[i]
                stack.offerLast(top)
            }
        }

        while (stack.isNotEmpty()) {
            var top = stack.pollLast()
            maxArea = maxOf(maxArea, heights[top] * (heights.size - top))
        }

        return maxArea
    }
}

/// 75. 颜色分类
class Solution75 {
    fun sortColors(nums: IntArray): Unit {
        var left = 0 // 从左往右第一个非0的数的下标
        var right = nums.size - 1 // 从右往左第一个非2的数的下标

        while (left < right) {
            while (left < nums.size && nums[left] == 0) left++
            while (right >= 0 && nums[right] == 2) right--

            if (left >= right) break

            // nums[left] 可能是 1 或 2
            // nums[right] 可能是 1 或 0
            if (nums[left] == 2) {
                if (nums[right] == 1) {
                    nums[left] = nums[right].also { nums[right] = nums[left] }
                    right--
                } else { // nums[right] == 0
                    nums[left] = nums[right].also { nums[right] = nums[left] }
                    left++
                    right--
                }
            } else { // nums[left] == 1
                if (nums[right] == 1) {
                    var moved = false
                    var r = right - 1
                    while (r > left) {
                        if (nums[r] == 2) {
                            nums[r] = nums[right].also { nums[right] = nums[r] }
                            right--
                            moved = true
                            break
                        }
                        r -= 1
                    }

                    var l = left + 1
                    while (l < right) {
                        if (nums[l] == 0) {
                            nums[l] = nums[left].also { nums[left] = nums[l] }
                            left++
                            moved = true
                            break
                        }
                        l += 1
                    }

                    if (!moved) {
                        break
                    }
                } else { // nums[right] == 0
                    nums[left] = nums[right].also { nums[right] = nums[left] }
                    left++
                }
            }
        }
    }

    fun sortColors_OnePointer(nums: IntArray): Unit {
        var index = 0
        for (i in nums.indices) {
            if (nums[i] == 0) {
                nums[i] = nums[index].also { nums[index] = nums[i] }
                index++
            }
        }
        index = nums.size - 1
        for (i in nums.indices.reversed()) {
            if (nums[i] == 2) {
                nums[i] = nums[index].also { nums[index] = nums[i] }
                index--
            }
        }
    }

    fun sortColors_TwoPointers(nums: IntArray): Unit {
        val n = nums.size
        var p0 = 0
        var p2 = n - 1
        var i = 0
        while (i <= p2) {
            while (i <= p2 && nums[i] == 2) {
                if (i != p2)
                    nums[i] = nums[p2].also { nums[p2] = nums[i] }
                p2--
            }
            if (nums[i] == 0) {
                if (i != p0) {
                    nums[i] = nums[p0].also { nums[p0] = nums[i] }
                }
                p0++
            }

            i++
        }
    }

    /// 计数排序
    fun sortColors_CountSort(nums: IntArray): Unit {
        var count0 = 0
        var count1 = 0
        var count2 = 0
        for (n in nums) {
            when (n) {
                0 -> count0++
                1 -> count1++
                2 -> count2++
            }
        }
        for (i in 0 until count0) {
            nums[i] = 0
        }
        for (i in count0 until count0 + count1) {
            nums[i] = 1
        }
        for (i in count0 + count1 until count0 + count1 + count2) {
            nums[i] = 2
        }
    }

    fun sortColors_CountSort2(nums: IntArray): Unit {
        val count = intArrayOf(0, 0, 0)
        for (n in nums) {
            count[n]++
        }
        for (i in 0 until count[0]) {
            nums[i] = 0
        }
        for (i in count[0] until count[0] + count[1]) {
            nums[i] = 1
        }
        for (i in count[0] + count[1] until count[0] + count[1] + count[2]) {
            nums[i] = 2
        }
    }

    fun sortColors_QuickSortPartition(nums: IntArray): Unit {
        var pivot = 2
        var i = 0
        for (j in nums.indices) { // 下标i走的总是比j慢
            if (nums[j] < pivot) {
                if (i != j) nums[j] = nums[i].also { nums[i] = nums[j] }
                i++
            }
        }
        var end = i
        pivot = 1
        i = 0
        for (j in 0 until end) {
            if (nums[j] < pivot) {
                if (i != j) nums[j] = nums[i].also { nums[i] = nums[j] }
                i++
            }
        }
    }
}

/// 78. 子集
class Solution78 {
    fun subsets(nums: IntArray): List<List<Int>> {
        val results = mutableListOf<List<Int>>()
        fun dfs(index: Int, current: MutableList<Int>) {
            if (index == nums.size) {
                results.add(current.toList())
                return
            }
            // 不选
            dfs(index + 1, current)
            // 选
            current.add(nums[index])
            dfs(index + 1, current)
            current.removeAt(current.size - 1)
        }
        dfs(0, mutableListOf<Int>())
        return results
    }

    fun subsets_bits(nums: IntArray): List<List<Int>> {
        val results = mutableListOf<List<Int>>()
        val stateCount = 1 shl nums.size
        for (s in 0 until stateCount) {
            val set = mutableListOf<Int>()
            for (bit in 0 until nums.size) {
                if (s and (1 shl bit) != 0) {
                    set.add(nums[bit])
                }
            }
            results.add(set)
        }
        return results
    }
}

/// 90. 子集 II
class Solution90 {
    fun subsetsWithDup_(nums: IntArray): List<List<Int>> {
        val results = mutableSetOf<List<Int>>()
        val stateCount = 1 shl nums.size
        for (s in 0 until stateCount) {
            val set = mutableListOf<Int>()
            for (bit in 0 until nums.size) {
                if (s and (1 shl bit) != 0) {
                    set.add(nums[bit])
                }
            }
            set.sort()
            results.add(set)
        }
        return results.toList()
    }

    fun subsetsWithDup_BinaryStates(nums: IntArray): List<List<Int>> {
        nums.sort()
        val results = mutableListOf<List<Int>>()
        val stateCount = 1 shl nums.size
        for (s in 0 until stateCount) {
            val set = mutableListOf<Int>()
            var ignore = false
            for (bit in 0 until nums.size) {
                if (s and (1 shl bit) != 0) {
                    if (bit > 0 && (s shr (bit - 1)) and 1 == 0 && nums[bit] == nums[bit - 1]) {
                        // 选择了第 bit 个数，但是没有选择第 bit-1 个数，且这两个数相等。
                        ignore = true
                        break
                    }
                    set.add(nums[bit])
                }
            }
            if (!ignore) {
                results.add(set)
            }
        }
        return results
    }

    fun subsetsWithDup_Recall(nums: IntArray): List<List<Int>> {
        nums.sort()
        val results = mutableListOf<List<Int>>()

        fun dfs(index: Int, choosePre: Boolean, current: MutableList<Int>) {
            if (index == nums.size) {
                results.add(current.toList())
                return
            }
            // 不选
            dfs(index + 1, false, current)

            // 选
            // 选择了当前的数，且没选择前一个数，且这两个数相当，这这个分支可以丢弃，因为会有其它分支覆盖本分支的情况
            if (!choosePre && nums[index] == nums[index - 1]) {
                return
            }
            // 正常选择当前这个数
            current.add(nums[index])
            dfs(index + 1, true, current)
            current.removeAt(current.size - 1) // 遍历完选择当前数的情况，将当前数删除，回溯到之前的状态
        }

        dfs(0, true, mutableListOf<Int>())
        return results
    }
}

/// 829. 连续整数求和
class Solution829 {
    fun consecutiveNumbersSum(n: Int): Int {
        /*
        设连续 k 个正整数加和为 n，且第一个正整数为 x，则有
        (x + (x + k - 1)) * k / 2 = n
        (x + (x + k - 1)) * k = 2n
        (2x + k - 1) * k = 2n         => 已知 x >= 1, k >= 1, 所以 2x >= 2 => 2x - 1 >= 1 => 2x - 1 + k >= k + 1 => 2x - 1 + k > k => k*k < 2n (即所有k需要满足 k 的平方小于 2n)
        (2x + k - 1) = 2n / k => k必须是2n的约数，且存在 x 满足此表达式

        x = (2n / k - (k - 1)) / 2 => x必须为正整数，即 (2n / k - (k - 1)) 必须可以被2整除（最低二进制位为0）
        */
        var count = 0
        // k 从 1 开始
        var k = 1
        while (k * k < 2 * n) {
            if (2 * n % k != 0) {
                k++
                continue // 2n必须可以整除k, 即k必须是2n的约数
            }
            if ((2 * n / k - k + 1) and 1 == 0) count++

            k++
        }
        return count
    }
}

/// 124. 二叉树中的最大路径和
class Solution124 {
    var max = Int.MIN_VALUE
    val maxPathSumStartNode = mutableMapOf<TreeNode, Int>()
    fun travel(root: TreeNode?): Int {
        if (root == null) return 0

        val leftSum = maxOf(travel(root.left), 0)
        val rightSum = maxOf(travel(root.right), 0) // 0代表可以不选择右子树中对应的路径

        max = maxOf(max, root.`val` + leftSum + rightSum)

        return maxOf(leftSum, rightSum) + root.`val`
    }

    fun maxPathSum(root: TreeNode?): Int {
        travel(root)
        return max
    }
}

/// 226. 翻转二叉树
class Solution226 {
    fun invertTree(root: TreeNode?): TreeNode? {
        if (root == null) return root
        root.left = root.right.also { root.right = root.left }
        invertTree(root.left)
        invertTree(root.right)
        return root
    }
}

/// 94. 二叉树的中序遍历
class Solution94 {
    fun inorderTraversal(root: TreeNode?): List<Int> {
        val results = mutableListOf<Int>()
        fun traversal(root: TreeNode?) {
            if (root == null) return
            traversal(root.left)
            results.add(root.`val`)
            traversal(root.right)
        }
        traversal(root)
        return results
    }
}

/// 155. 最小栈
class MinStack() {
    class Node(val value: Int, var min: Int)

    val datas = LinkedList<Node>()

    fun push(value: Int) {
        var min = datas.peekLast()?.min ?: value
        min = minOf(min, value)
        datas.offer(Node(value, min))
    }

    fun pop() {
        datas.pollLast()
    }

    fun top(): Int {
        return datas.peekLast().value
    }

    fun getMin(): Int {
        return datas.peekLast()?.min ?: -Int.MIN_VALUE
    }

}

/// 101. 对称二叉树
class Solution101 {
    // 考虑中序遍历，然后检测结果数组是否是对称的
    fun isSymmetric(root: TreeNode?): Boolean {
        if (root?.left?.`val` != root?.right?.`val`) return false

        val list = mutableListOf<Int>()
        fun travel(root: TreeNode?) {
            if (root == null) return

            if (root.left != null && root.right != null) {
                travel(root.left)
                list.add(root.`val`)
                travel(root.right)
            } else if (root.left == null && root.right != null) {
                list.add(Int.MIN_VALUE)
                list.add(root.`val`)
                travel(root.right)
            } else if (root.left != null && root.right == null) {
                travel(root.left)
                list.add(root.`val`)
                list.add(Int.MIN_VALUE)
            } else {
                list.add(root.`val`)
            }
        }
        travel(root)
        var left = 0
        var right = list.size - 1
        while (left < right) {
            if (list[left] != list[right]) {
                return false
            }
            left++
            right--
        }
        return true
    }
}

class Solution101_Recursive {
    fun check(p: TreeNode?, q: TreeNode?): Boolean {
        if (p == null && q == null) return true
        if (p == null || q == null) return false
        return p.`val` == q.`val` && check(p.left, q.right) && check(p.right, q.left)
    }

    fun isSymmetric(root: TreeNode?): Boolean {
        return check(root?.left, root?.right)
        // return check(root, root)
    }
}

class Solution101_Iteration {
    fun isSymmetric(root: TreeNode?): Boolean {
        if (root == null) return true

        // 左子树 和 右子树 镜像对称
        val queue = LinkedList<Pair<TreeNode?, TreeNode?>>()
        queue.offer(Pair(root, root))
        while (queue.isNotEmpty()) {
            val (left, right) = queue.poll()

            if (left == null && right == null) continue
            if (left == null || right == null) return false
            if (left!!.`val` != right.`val`) return false

            queue.offer(Pair(left!!.left, right!!.right))
            queue.offer(Pair(left!!.right, right!!.left))
        }

        return true
    }
}

/// 169. 多数元素
class Solution169 {
    fun majorityElement(nums: IntArray): Int {
        var counts = mutableMapOf<Int, Int>()

        for (n in nums) {
            counts[n] = 0
        }
        for (n in nums) {
            counts[n] = counts[n]!! + 1
            if (counts[n]!! > nums.size / 2) return n
        }
        return Int.MIN_VALUE
    }

    fun majorityElement_BM(nums: IntArray): Int {
        var count = 0
        var candidate: Int? = null
        for (n in nums) {
            if (count == 0) candidate = n
            count += (if (n == candidate) 1 else -1)
        }
        return candidate!!
    }
}

/// 543. 二叉树的直径
class Solution543 {
    fun diameterOfBinaryTree(root: TreeNode?): Int {
        var max = 0
        fun dfs(root: TreeNode?): Int {
            if (root == null) return 0

            val leftmax = dfs(root.left)
            val rightmax = dfs(root.right)

            max = maxOf(max, leftmax + rightmax + 1)
            return maxOf(leftmax, rightmax) + 1
        }
        dfs(root)
        return max - 1
    }
}

/// 104. 二叉树的最大深度
class Solution104 {
    fun maxDepth(root: TreeNode?): Int {
        var max = 0
        fun dfs(root: TreeNode?, depth: Int) {
            if (root == null) {
                max = maxOf(max, depth)
                return
            }
            dfs(root.left, depth + 1)
            dfs(root.right, depth + 1)
        }
        dfs(root, 0)
        return max
    }
}

/// 76. 最小覆盖子串
class Solution76 {
    fun minWindow(s: String, t: String): String {
        if (t.isEmpty()) return ""

        val need = mutableMapOf<Char, Int>()
        for (c in t) {
            need[c] = need.getOrDefault(c, 0) + 1
        }

        val window = mutableMapOf<Char, Int>()
        var left = 0
        var right = 0
        var valid = 0

        // 结果子串信息
        var start = 0
        var len = Int.MAX_VALUE

        while (right < s.length) {
            val c = s[right]
            right++
            if (need.containsKey(c)) {
                window[c] = window.getOrDefault(c, 0) + 1
                if (need[c]!! == window[c]!!) {
                    valid++
                }
            }

            while (valid == need.size) {
                if (right - left < len) {
                    start = left
                    len = right - left
                }
                val c = s[left]
                if (need.containsKey(c)) {
                    if (window[c]!! == need[c]!!) {
                        valid--
                    }
                    // 先做判断，再减去数量
                    window[c] = window[c]!! - 1
                }
                left++
            }
        }

        if (len == Int.MAX_VALUE) {
            return ""
        }
        return s.substring(start..(start + len - 1))
    }
}

/// 239. 滑动窗口最大值
class Solution239 {
    fun maxSlidingWindow_(nums: IntArray, k: Int): IntArray {
        val n = nums.size
        val results = IntArray(n - (k - 1))
        val priorityQueue = PriorityQueue<Int> { n1, n2 -> n2 - n1 }

        for (i in 0 until k - 1) {
            priorityQueue.add(nums[i])
        }
        for (i in k - 1 until n) {
            priorityQueue.add(nums[i])
            results[i - (k - 1)] = priorityQueue.peek()
            priorityQueue.remove(nums[i - (k - 1)])
        }
        return results
    }

    fun maxSlidingWindow_PQ(nums: IntArray, k: Int): IntArray {
        val n = nums.size
        val results = IntArray(n - (k - 1))
        val priorityQueue = PriorityQueue<Pair<Int, Int>> { n1, n2 -> n2.first - n1.first }

        for (i in 0 until k - 1) {
            priorityQueue.add(Pair(nums[i], i))
        }
        for (i in k - 1 until n) {
            priorityQueue.add(Pair(nums[i], i))
            while (priorityQueue.peek().second < i - (k - 1)) priorityQueue.poll()
            // 从 nums 中第 i-(k-1)..i 共 k 个数中选择最大的值
            results[i - (k - 1)] = priorityQueue.peek()!!.first
        }
        return results
    }

    fun maxSlidingWindow_PQ_OP(nums: IntArray, k: Int): IntArray {
        val n = nums.size
        val results = IntArray(n - (k - 1))
        val queue = LinkedList<Int>()

        for (i in 0 until k - 1) {
            while (queue.isNotEmpty() && nums[i] >= nums[queue.peekLast()!!]) {
                queue.pollLast()
            }
            queue.offerLast(i)
        }
        for (i in k - 1 until n) {
            while (queue.isNotEmpty() && nums[i] >= nums[queue.peekLast()!!]) {
                queue.pollLast()
            }
            queue.offerLast(i)
            while (queue.peekFirst() < i - (k - 1)) {
                queue.pollFirst()
            }
            // 从 nums 中第 i-(k-1)..i 共 k 个数中选择最大的值
            results[i - (k - 1)] = nums[queue.peekFirst()!!]
        }
        return results
    }
}

/// 929. 独特的电子邮件地址
class Solution929 {
    fun numUniqueEmails(emails: Array<String>): Int {
        val realEmails = mutableSetOf<String>()
        for (email in emails) {
            val indexOfAtSign = email.indexOf('@')
            if (indexOfAtSign != -1) {
                var prefix = email.substring(0 until indexOfAtSign)
                var suffix = email.substring(indexOfAtSign)
                val indexOfPlusSign = prefix.indexOf('+')
                if (indexOfPlusSign != -1) {
                    prefix = prefix.substring(0 until indexOfPlusSign)
                }
                prefix = prefix.replace(".", "")
                realEmails.add(prefix + suffix)
            }
        }
        return realEmails.size
    }
}

/// 47. 全排列 II
class Solution47 {
    fun permuteUnique(nums: IntArray): List<List<Int>> {
        val results = mutableListOf<List<Int>>()
        val used = BooleanArray(nums.size)
        fun dfs(path: LinkedList<Int>) {
            if (path.size == nums.size) {
                results.add(path.toList())
                return
            }
            var lastUnused: Int? = null
            for (i in nums.indices) {
                if (!used[i]) {
                    if (nums[i] == lastUnused) {
                        continue // 本次迭代使用过了 nums[i]
                    }
                    lastUnused = nums[i]

                    path.offerLast(nums[i])
                    used[i] = true
                    dfs(path)
                    used[i] = false
                    path.pollLast()
                }
            }
        }
        nums.sort()
        dfs(LinkedList<Int>())
        return results
    }
}

/// 40. 组合总和 II
class Solution40 {
    fun combinationSum2(candidates: IntArray, target: Int): List<List<Int>> {
        if (candidates.sum() < target) return emptyList()

        val n = candidates.size
        val results = mutableListOf<List<Int>>()
        fun dfs(selection: LinkedList<Int>, target: Int, from: Int) {
            for (i in from until n) {
                if (i > from && candidates[i] == candidates[i - 1]) {
                    continue
                }
                if (target == candidates[i]) {
                    selection.offerLast(candidates[i])
                    results.add(selection.toList())
                    selection.pollLast()
                    break
                } else if (target > candidates[i]) {
                    selection.offerLast(candidates[i])
                    dfs(selection, target - candidates[i], i + 1)
                    selection.pollLast()
                } else {
                    break
                }
            }
        }

        candidates.sort()
        dfs(LinkedList<Int>(), target, 0)
        return results
    }
}

/// 312. 戳气球
class Solution312 {
    fun maxCoins_Recursive(nums: IntArray): Int {
        val n = nums.size
        val values = IntArray(n + 2) { i ->
            var value = 1
            if (i >= 1 && i <= n) {
                value = nums[i - 1]
            }
            value
        }

        val mem = Array(n + 2) { IntArray(n + 2) { -1 } }

        /// 戳破 i 和 j 之间的气球，注意不会戳位置 i 和 j 的气球，i 和 j 仅仅作为一个边界使用。
        fun poke(i: Int, j: Int): Int {
            if (i >= j - 1) {
                return 0
            }
            if (mem[i][j] != -1) {
                return mem[i][j]
            }
            for (k in i + 1 until j) {
                // 戳破区间 (i,j) 内的气球所得金币最大值为：戳破 (i,k) 内气球得到的金币值 + 戳破 (k,j) 内气球得到的金币值 + 戳破气球 k 得到的金币值(因为 i~k 和 k~j 内的气球都被戳破了，
                // 所以，戳破气球 k 得到的金币值为 values[i] * values[k] * values[j]， 即气球 k 左边为球球 i，右边为气球 j)
                // !! 这个过程中始终不会去动气球 i 和 j
                mem[i][j] =
                    maxOf(mem[i][j], poke(i, k) + poke(k, j) + values[i] * values[k] * values[j])
            }
            return mem[i][j]
        }
        // 假设按任意顺序去戳破气球，然后统计所有顺序下获得的金币值，取最大保留。
        //
        // 假设找到了最大金币值对应的戳气球顺序，在戳破 values 中最后一个气球 k 时，比如已经将 values[1 until k] 和 values[k until n+2] 这两个区间内的所有气球戳破了，对应已经得到了 coinCount[1,k] 和 coinCount[k,n+2],
        // 再加上戳破气球 k 的金币数 values[0] * values[k] * values[n+1] 即为 coinCount[0,n+1]
        // 将以上过程递归地应用到开区间 (0, k) 和 (k, n+1) 中.
        return poke(0, n + 1)
    }

    fun maxCoins_DP(nums: IntArray): Int {
        val n = nums.size
        val values = IntArray(n + 2) { i ->
            var value = 1
            if (i >= 1 && i <= n) {
                value = nums[i - 1]
            }
            value
        }

        // dp[i][j]代表戳破开区间 (i, j) 中所有气球（开区间，所以不包括气球i和j）获得的金币数
        val dp = Array(n + 2) { IntArray(n + 2) }
        for (len in 3..(n + 2)) {
            for (start in 0..(n + 2 - len)) {
                val end = start + len - 1
                for (middle in start + 1 until end) {
                    dp[start][end] = maxOf(
                        dp[start][end],
                        dp[start][middle] + dp[middle][end] + values[start] * values[middle] * values[end]
                    )
                }
            }
        }
        return dp[0][n + 1]
    }
}

/// 1000. 合并石头的最低成本
class Solution1000 {
    /// 搓石头
    fun mergeStones(stones: IntArray, k: Int): Int {
        val MAX = 99999999
        val n = stones.size
        if (n == 1) return 0
        // 合并 m 次后，m > 0，合并1次石头堆数减少 K-1 堆，最终剩下 n - (K - 1) * m = 1 才能保证最终合并成了1堆
        // 只要找得到整数 m 则可行。
        if ((n - 1) % (k - 1) != 0) return -1

        val sum = IntArray(n + 1) // 前缀和
        for (i in 1..n) {
            sum[i] = sum[i - 1] + stones[i - 1]
        }

        // dp[i][j][k] 代表将 stones[i] ~ stones[j] 合并成 k 堆的最小成本
        val dp = Array(n + 1) { Array(n + 1) { IntArray(k + 1) } }
        for (i in 1..n) {
            for (j in i..n) {
                for (m in 2..k) dp[i][j][m] = MAX
            }
            dp[i][i][1] = 0 // 每一堆石头合并成1堆的成本为0
        }

        for (len in 2..n) { // 枚举区间长度
            for (i in 1..(n + 1 - len)) { // 枚举区间起点
                val j = i + len - 1 // 确定区间终点
                for (m in 2..k) { // 枚举合并堆数
                    for (p in i until j step k - 1) { // 枚举分界点
                        // 找到一个分界点 p, 让 dp[i][p][1] + dp[p + 1][j][m - 1]) 最小
                        // 将 [i..p] 合成 1 堆，然后将 [p+1, j] 合成 k-1 堆
                        dp[i][j][m] =
                            minOf(dp[i][j][m], dp[i][p][1] + dp[p + 1][j][m - 1])
                    }
                }
                // dp[i][j][k] 对应 stones[i-1..j-1] 合并成 k 堆的最小成本
                // sum[j] 为 stones[0..j-1] 的和
                // sum[i-1] 为 stones[0..i-2] 的和
                dp[i][j][1] = dp[i][j][k] + sum[j] - sum[i - 1] // 将 K 堆合成 1 堆
            }
        }
        return dp[1][n][1]
    }
}


/// 1000. 合并石头的最低成本
class Solution1000_20220605 {
    /// 搓石头
    fun mergeStones(stones: IntArray, k: Int): Int {
        val MAX = 99999999
        val n = stones.size
        if (n == 1) return 0
        // 合并 m 次后，m > 0，合并1次石头堆数减少 K-1 堆，最终剩下 n - (K - 1) * m = 1 才能保证最终合并成了1堆
        // 只要找得到整数 m 则可行。
        if ((n - 1) % (k - 1) != 0) return -1

        val sum = IntArray(n + 1) // 前缀和
        for (i in 1..n) {
            sum[i] = sum[i - 1] + stones[i - 1]
        }

        // dp[i][j][k] 代表将 stones[i] ~ stones[j] 合并成 k 堆的最小成本
        val dp = Array(n + 1) { Array(n + 1) { IntArray(k + 1) } }
        for (i in 1..n) {
            for (j in i..n) {
                for (m in 2..k) dp[i][j][m] = MAX
            }
            dp[i][i][1] = 0 // 每一堆石头合并成1堆的成本为0
        }

        for (len in 2..n) { // 枚举区间长度
            for (i in 1..(n + 1 - len)) { // 枚举区间起点
                val j = i + len - 1 // 确定区间终点
                for (m in 2..k) { // 枚举合并堆数
                    for (p in i until j step k - 1) { // 枚举分界点
                        // 找到一个分界点 p, 让 dp[i][p][1] + dp[p + 1][j][m - 1]) 最小
                        // 将 [i..p] 合成 1 堆，然后将 [p+1, j] 合成 k-1 堆
                        dp[i][j][m] =
                            minOf(dp[i][j][m], dp[i][p][1] + dp[p + 1][j][m - 1])
                    }
                }
                // dp[i][j][k] 对应 stones[i-1..j-1] 合并成 k 堆的最小成本
                // sum[j] 为 stones[0..j-1] 的和
                // sum[i-1] 为 stones[0..i-2] 的和
                dp[i][j][1] = dp[i][j][k] + sum[j] - sum[i - 1] // 将 K 堆合成 1 堆
            }
        }
        return dp[1][n][1]
    }

    fun mergeStones_DP_OP(stones: IntArray, k: Int): Int {
        val n = stones.size
        if ((n - 1) % (k - 1) != 0) return -1

        /// 前缀和
        val sum = IntArray(n + 1)
        for (i in 1..n) {
            sum[i] = sum[i - 1] + stones[i - 1]
        }

        // dp[i][j] 代表将石堆 i 到 j 尽可能多地合并所需的最小成本
        // dp[i][j] = 0 如果 j-i+1 < k，因为按照题意一次只能将 k 个石堆合成 1 堆。
        //
        // dp[i][j] = minOf(dp[i][j], dp[i][p] + dp[p+1][j])，找到一个 p 让取值最小。
        // dp[i][j] += sum(i, j) 如果 ((j-i+1)-1) % (k - 1) == 0 => (j-i) % (k-1) == 0 即第i~j堆石头可以合并成1堆
        val dp = Array(n + 1) { IntArray(n + 1) }
        for (len in k..n) { // 枚举区间长度
            for (i in 1..(n - (len - 1))) { // 枚举区间起点
                val j = i + (len - 1) // 枚举区间终点
                dp[i][j] = Int.MAX_VALUE
                for (p in i until j step k - 1) {
                    dp[i][j] = minOf(dp[i][j], dp[i][p] + dp[p + 1][j])
                }
                // 此时已经找到将第i~j堆是都合并成1堆和k-1堆的最小成本，如果可以将第i~j堆合并成1堆的话，则直接再进行一次合并
                if ((j - i) % (k - 1) == 0) {
                    dp[i][j] += sum[j] - sum[i - 1]
                }
            }
        }
        return dp[1][n]
    }
}


/// 478. 在圆内随机生成点
class Solution478(val radius: Double, val x_center: Double, val y_center: Double) {

    var rand = Random()

    fun randPoint(): DoubleArray {
        while (true) {
            val x = rand.nextDouble() * 2 * radius - radius
            val y = rand.nextDouble() * 2 * radius - radius
            if (x * x + y * y <= radius * radius) {
                return doubleArrayOf(x_center + x, y_center + y)
            }
        }
    }
}

/// 79. 单词搜索
class Solution79 {
    fun exist(board: Array<CharArray>, word: String): Boolean {
        val rowCount = board.size
        val colCount = board[0].size

        val locations = LinkedList<Triple<Int, Int, Int>>()
        // 找到所有word首字母在board中的位置
        for (r in 0 until rowCount) {
            for (c in 0 until colCount) {
                if (board[r][c] == word[0]) {
                    locations.offerLast(Triple(r, c, 0))
                }
            }
        }

        // word 只有一个字符
        if (word.length == 1) return locations.isNotEmpty()

        // word.length > 1
        val directions = listOf(Pair(1, 0), Pair(-1, 0), Pair(0, 1), Pair(0, -1))
        val visited = Array(rowCount) { BooleanArray(colCount) }

        fun dfs(row: Int, col: Int, index: Int): Boolean {
            if (!(row in 0 until rowCount) || !(col in 0 until colCount) || visited[row][col]) return false
            if (index == word.length - 1) return board[row][col] == word[index]
            if (board[row][col] != word[index]) return false

            visited[row][col] = true
            var found = false
            for ((dr, dc) in directions) {
                val r = row + dr
                val c = col + dc
                if (dfs(r, c, index + 1)) {
                    found = true
                    break
                }
            }
            visited[row][col] = false
            return found
        }

        while (locations.isNotEmpty()) {
            val (row, col, index) = locations.pollFirst()
            visited.forEach {
                for (i in it.indices) {
                    it[i] = false
                }
            }
            if (dfs(row, col, index)) {
                return true
            }
        }


        return false
    }
}

/// 731. 我的日程安排表 II
class MyCalendarTwo731() {
    val calendar = TreeMap<Int, Int>()

    fun book(start: Int, end: Int): Boolean {
        calendar[start] = calendar.getOrDefault(start, 0) + 1
        calendar[end] = calendar.getOrDefault(end, 0) - 1

        var count = 0
        for (entry in calendar.entries) {
            count += entry.value
            if (count >= 3) {
                calendar[start] = calendar[start]!! - 1
                calendar[end] = calendar[end]!! + 1
                if (calendar[start] == 0) {
                    calendar.remove(start)
                }
                return false
            }
        }
        return true
    }
    /**
     * Your MyCalendarTwo object will be instantiated and called as such:
     * var obj = MyCalendarTwo()
     * var param_1 = obj.book(start,end)
     */
}

/// 732. 我的日程安排表 III
class MyCalendarThree732() {
    val calendar = TreeMap<Int, Int>()

    fun book(start: Int, end: Int): Int {
        calendar[start] = calendar.getOrDefault(start, 0) + 1
        calendar[end] = calendar.getOrDefault(end, 0) - 1

        var maxCount = 0
        var count = 0
        for (entry in calendar.entries) {
            count += entry.value
            maxCount = maxOf(maxCount, count)
        }

        return maxCount
    }
    /**
     * Your MyCalendarThree object will be instantiated and called as such:
     * var obj = MyCalendarThree()
     * var param_1 = obj.book(start,end)
     */
}

/// 1229. 安排会议日程
class Solution1229 {
    fun minAvailableDuration(
        slots1: Array<IntArray>,
        slots2: Array<IntArray>,
        duration: Int
    ): List<Int> {
        val n1 = slots1.size
        val n2 = slots2.size
        slots1.sortBy { a -> a[0] }
        slots2.sortBy { a -> a[0] }
        var i1 = 0
        var i2 = 0
        while (i1 < n1 && i2 < n2) {
            val start1 = slots1[i1][0]
            val end1 = slots1[i1][1]
            val start2 = slots2[i2][0]
            val end2 = slots2[i2][1]

            // 判断是否相交，判断相交时长是否符合要求
            if (start1 <= end2 && end1 >= start2) {
                val start = maxOf(start1, start2)
                val end = minOf(end1, end2)
                val interval = end - start
                if (interval >= duration) {
                    return listOf(start, start + duration)
                }
            }

            if (end1 == end2) {
                i1 += 1
                i2 += 1
            } else if (end2 < end1) {
                i2 += 1
            } else {
                i1 += 1
            }
        }
        return emptyList()
    }
}

/// 875. 爱吃香蕉的珂珂
class Solution875 {
    fun minEatingSpeed(piles: IntArray, h: Int): Int {
        piles.sort()
        val n = piles.size

        var min = 1
        var max = piles[n - 1]
        var speed = 0
        var hours = 0

        while (min <= max) {
            speed = (min + max) ushr 1

            hours = 0
            for (pile in piles) {
                hours += Math.ceil(pile * 1.0 / speed).toInt()
            }

            if (hours <= h) {
                var hours1 = 0
                for (pile in piles) {
                    hours1 += Math.ceil(pile * 1.0 / (speed - 1)).toInt()
                }
                if (hours1 > h) {
                    break
                }
                // hours1 <= h
                // 速度太快
                max = speed - 1
            } else { // hours > h
                // 速度太慢
                min = speed + 1
            }
        }

        return speed
    }
}

/// 45. 跳跃游戏 II
class Solution45 {
    fun jump(nums: IntArray): Int {
        val n = nums.size
        var position = n - 1
        var jumpCount = 0
        while (position > 0) {
            for (i in 0 until position) {
                if (i + nums[i] >= position) {
                    position = i
                    jumpCount++
                    break
                }
            }
        }
        return jumpCount
    }

    fun jump_Forward(nums: IntArray): Int {
        val n = nums.size
        var from = 0
        var to = 0
        var jumpCount = 0
        while (to < n - 1) {
            jumpCount++

            var maxReachable = to
            for (i in from..to) {
                if (i + nums[i] > maxReachable) {
                    maxReachable = i + nums[i]
                    if (maxReachable >= n - 1) {
                        return jumpCount
                    }
                }
            }
            from = to + 1
            to = maxReachable
        }
        return jumpCount
    }
}

/// 413. 等差数列划分
class Solution413 {
    fun numberOfArithmeticSlices(nums: IntArray): Int {
        if (nums.size < 3) return 0
        var left = 0
        var right = 0
        var count = 0
        while (right <= nums.size) {
            val size = right - left + 1
            if (size < 3) {
                ++right
            } else {
                val d = nums[left + 1] - nums[left]
                if (d != nums.getOrElse(right) { 9999 } - nums[right - 1]) {
                    val size = right - left
                    /*if (d == 0) {
                        count += size - 2
                    } else {*/
                    count += (size - 2) * (size - 1) / 2
                    //}
                    left = right - 1
                } else {
                    ++right
                }
            }
        }
        return count
    }
}

/// 413. 等差数列划分
class Solution413_20220609 {
    fun numberOfArithmeticSlices(nums: IntArray): Int {
        val n = nums.size
        if (n < 3) return 0

        var d = nums[0] - nums[1]
        var count = 0
        var sum = 0
        for (i in 2 until n) {
            if (nums[i - 1] - nums[i] == d) {
                count += 1
            } else {
                count = 0
                d = nums[i - 1] - nums[i]
            }
            sum += count
        }
        return sum
    }
}

/// 1037. 有效的回旋镖
class Solution1037 {
    fun isBoomerang(points: Array<IntArray>): Boolean {
        if (points.size != 3) return false

        // 斜率 = (y0 - y1) / (x0 - x1) = (y2 - y1) / (x2 - x1)

        val x0 = points[0][0]
        val y0 = points[0][1]
        val x1 = points[1][0]
        val y1 = points[1][1]
        val x2 = points[2][0]
        val y2 = points[2][1]

        return (x0 - x1) * (y2 - y1) != (y0 - y1) * (x2 - x1)
    }
}

/// 497. 非重叠矩形中的随机点
class Solution497(val rects: Array<IntArray>) {

    val rand = Random()
    var areas = mutableListOf<Long>()
    var areasSum:Long = 0

    init {
        for (rect in rects) {
            val x0 = rect[0].toLong()
            val y0 = rect[1].toLong()
            val x1 = rect[2].toLong()
            val y1 = rect[3].toLong()
            val area = (x1 - x0 + 1) * (y1 - y0 + 1) // 计算在矩形内的整数点数
            areas.add(area)
            areasSum += area
        }
    }

    fun check(point: IntArray): Boolean {
        for (rect in rects) {
            if (rect[0] <= point[0] && point[0] <= rect[2] && rect[1] <= point[1] && point[1] <= rect[3]) {
                return true
            }
        }
        return false
    }

    private fun randomPointInRect(rect: IntArray): IntArray {
        val x0 = rect[0]
        val y0 = rect[1]
        val x1 = rect[2]
        val y1 = rect[3]
        return intArrayOf(x0 + rand.nextInt(x1 - x0 + 1), y0 + rand.nextInt(y1 - y0 + 1))
    }

    fun pick(): IntArray {
        val targetSum = (rand.nextDouble() * areasSum).toLong()
        var sum: Long = 0
        var targetRectIndex = 0
        for (i in areas.indices) {
            val low = sum
            sum += areas[i]
            if (low <= targetSum && targetSum < sum) {
                targetRectIndex = i
            }
        }
        return randomPointInRect(rects[targetRectIndex])
    }

    /**
     * Your Solution object will be instantiated and called as such:
     * var obj = Solution(rects)
     * var param_1 = obj.pick()
     */
}

/// 91. 解码方法
class Solution91 {
    fun check(c: Char): Boolean {
        val n = c - '0'
        return 0 < n && n <= 26
    }
    fun check(c: Char, d: Char): Boolean {
        if (c == '0') return false
        val n = (c - '0') * 10 + (d - '0')
        return 0 < n && n <= 26
    }
    fun numDecodings(s: String): Int {
        // dp[i]代表字符串s前i个字符组词的子串的解码方法数
        val dp = IntArray(s.length + 1)
        dp[0] = 1
        dp[1] = if (check(s[0])) 1 else 0
        for (count in 2 until s.length + 1) {
            var d = s[count - 1]
            if (check(d)) {
                dp[count] += dp[count - 1]
            }
            var c = s[count - 2]
            if (check(c, d)) {
                dp[count] += dp[count - 2]
            }
        }
        return dp[s.length]
    }
}

/// 497. 非重叠矩形中的随机点
class Solution497(val rects: Array<IntArray>) {

    val rand = Random()
    var areas = mutableListOf<Long>()
    var areasSum:Long = 0

    init {
        for (rect in rects) {
            val x0 = rect[0].toLong()
            val y0 = rect[1].toLong()
            val x1 = rect[2].toLong()
            val y1 = rect[3].toLong()
            val area = (x1 - x0 + 1) * (y1 - y0 + 1) // 计算在矩形内的整数点数
            areas.add(area)
            areasSum += area
        }
    }

    fun check(point: IntArray): Boolean {
        for (rect in rects) {
            if (rect[0] <= point[0] && point[0] <= rect[2] && rect[1] <= point[1] && point[1] <= rect[3]) {
                return true
            }
        }
        return false
    }

    private fun randomPointInRect(rect: IntArray): IntArray {
        val x0 = rect[0]
        val y0 = rect[1]
        val x1 = rect[2]
        val y1 = rect[3]
        return intArrayOf(x0 + rand.nextInt(x1 - x0 + 1), y0 + rand.nextInt(y1 - y0 + 1))
    }

    fun pick(): IntArray {
        val targetSum = (rand.nextDouble() * areasSum).toLong()
        var sum: Long = 0
        var targetRectIndex = 0
        for (i in areas.indices) {
            val low = sum
            sum += areas[i]
            if (low <= targetSum && targetSum < sum) {
                targetRectIndex = i
            }
        }
        return randomPointInRect(rects[targetRectIndex])
    }

    /**
     * Your Solution object will be instantiated and called as such:
     * var obj = Solution(rects)
     * var param_1 = obj.pick()
     */
}

/// 139. 单词拆分
class Solution139 {
    fun wordBreak_(s: String, wordDict: List<String>): Boolean {
        val wordSet = wordDict.toSet()
        // 我们定义 dp[i] 表示字符串 s 前 i 个字符组成的字符串 s[0..i-1] 是否能被空格拆分成若干个字典中出现的单词。
        val dp = BooleanArray(s.length + 1)
        dp[0] = true
        for (i in 1..s.length) {
            for (j in 0 until i) {
                if (dp[j] && wordSet.contains(s.substring(j, i))) {
                    dp[i] = true
                    break
                }
            }
        }
        return dp[s.length]
    }

    fun wordBreak(s: String, wordDict: List<String>): Boolean {
        val wordSet = wordDict.toSet()
        // dp[i]代表字符串s中前i个字符是否能用字典中的单词组成
        val dp = BooleanArray(s.length + 1)
        dp[0] = true // 空字符串可以用字典中的"空"组成
        for (i in 1 until s.length + 1) {
            for (j in 0 until i) {
                if (dp[j] && wordSet.contains(s.substring(j, i))) {
                    dp[i] = true
                    break
                }
            }
        }
        return dp[s.length]
    }
}

/// 730. 统计不同回文子序列
class Solution730 {
    fun countPalindromicSubsequences(s: String): Int {
        // val m = (1e9+7).toInt()
        val m = 1000000007
        val n = s.length
        // dp[x][i][j] 代表在s[i..j]中以字符c开头和结尾的回文子序列个数
        val dp = Array(4) { Array(n) { IntArray(n) }}
        for (i in 0 until n) {
            dp[s[i]-'a'][i][i] = 1
        }

        for (i in n-2 downTo 0) {
            for (j in i+1 until n) {
                for (x in 0 until 4) {
                    val c = ('a' + x).toChar()
                    val iIsC = s[i] == c
                    val jIsC = s[j] == c
                    if (iIsC && jIsC) {
                        dp[x][i][j] = 2
                        for (xx in 0 until 4) {
                            dp[x][i][j] = (dp[x][i][j] + dp[xx][i + 1][j - 1]) % m
                        }
                    } else if (iIsC && !jIsC) {
                        dp[x][i][j] = dp[x][i][j - 1]
                    } else if (!iIsC && jIsC) {
                        dp[x][i][j] = dp[x][i + 1][j]
                    } else {
                        dp[x][i][j] = dp[x][i + 1][j - 1]
                    }
                }
            }
        }

        var sum = 0
        for (x in 0 until 4) {
            sum += dp[x][0][n - 1]
            sum %= m
        }

        return sum
    }
}