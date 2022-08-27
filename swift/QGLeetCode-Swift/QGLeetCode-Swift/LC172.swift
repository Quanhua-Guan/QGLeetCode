//
//  LC172.swift
//  QGLeetCode-Swift
//
//  Created by 宇园 on 2022/8/28.
//

import Foundation

class LC172 {
    class Solution {
        func trailingZeroes(_ n: Int) -> Int {
            var n = n
            var cnt = 0
            while n > 0 {
                n /= 5
                cnt += n
            }            
            return cnt
        }
    }
}
