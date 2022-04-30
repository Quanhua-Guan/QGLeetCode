package com.cqmh.qgleetcode

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import java.util.*

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        Solution8().myAtoi("2147483648");
    }
}

// 两数之和
class Solution1 {
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

class ListNode(var `val`: Int) {
    var next: ListNode? = null
}

// 两数相加
class Solution2 {
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

        var charLists = List(numRows){LinkedList<Char>()}

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
                if (index == 0)  {
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