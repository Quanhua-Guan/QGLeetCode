//
//  LC1781.swift
//  QGLeetCode-Swift
//
//  Created by Quanhua on 2022/12/12.
//

import Foundation

// 1781. 所有子字符串美丽值之和

class LC1781 {
    
    class Solution {
        func beautySum(_ s: String) -> Int {
            var sum = 0
            let asciiA = Character("a").asciiValue!
            
            var count = [Int].init(repeating: 0, count: 26)
            for i in 0..<s.count {
                var maxi = 0
                
                for ii in count.indices {
                    count[ii] = 0
                }
                
                for j in i..<s.count {
                    let c = s[s.index(s.startIndex, offsetBy: j)]
                    let asciiValue = c.asciiValue!
                    let diff = Int(asciiValue - asciiA)
                    count[diff] += 1
                    maxi = max(maxi, count[diff])
                    
                    var mini = s.count
                    for cnt in count {
                        if cnt > 0 {
                            mini = min(mini, cnt)
                        }
                    }
                    
                    sum += maxi - mini
                }
            }
            
            return sum
        }
    }
    
}
