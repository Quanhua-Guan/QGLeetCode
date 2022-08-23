import Foundation

class Solution {
    func movesToChessboard(_ board: [[Int]]) -> Int {
        let n = board.count
        
        func getMask(_ index: Int) -> (Int, Int) {
            var colMask = 0
            var rowMask = 0
            for i in 0..<n {
                rowMask |= board[index][i] << i
                colMask |= board[i][index] << i
            }
            return (rowMask, colMask)
        }
        
        let (rowMask, colMask) = getMask(0)
        let reverseRowMask = ((1 << n) - 1) ^ rowMask
        let reverseColMask = ((1 << n) - 1) ^ colMask
        
        var rowMaskCount = 0
        var colMaskCount = 0
        for i in 0..<n {
            let (curRowMask, curColMask) = getMask(i)
            if curRowMask != rowMask && curRowMask != reverseRowMask {
                return -1
            } else if curRowMask == rowMask {
                rowMaskCount += 1
            }
            
            if curColMask != colMask && curColMask != reverseColMask {
                return -1
            } else if curColMask == colMask {
                colMaskCount += 1
            }
        }
        
        func switchCount(_ mask: Int, _ maskCount: Int) -> Int {
            let bit1Count = mask.nonzeroBitCount
            if (n & 1 == 1) { // 奇数
                if abs(bit1Count - n / 2) != 1 || abs(maskCount - n / 2) != 1 {
                    return -1
                }
                
                if (bit1Count == n / 2) {
                    // 0开头和结尾, 01010, 找到奇数位上0的个数 = (奇数位的位数 - 奇数位上1的位数)
                    return n / 2 - (mask & 0xAAAAAAAA).nonzeroBitCount
                } else {
                    // 1开头和结尾, 10101, 同理.
                    return (n + 1) / 2 - (mask & 0x55555555).nonzeroBitCount
                }
            } else { // 偶数
                if bit1Count != n / 2 || maskCount != n / 2 {
                    return -1
                }
                
                let count1AtFirst = n / 2 - (mask & 0xAAAAAAAA).nonzeroBitCount
                let count0AtFirst = n / 2 - (mask & 0x55555555).nonzeroBitCount
                return min(count1AtFirst, count0AtFirst)
            }
        }
        
        let rowSwitchCount = switchCount(rowMask, rowMaskCount)
        if rowSwitchCount == -1 { return -1 }
        
        let colSwitchCount = switchCount(colMask, colMaskCount)
        if colSwitchCount == -1 { return -1 }
        
        return rowSwitchCount + colSwitchCount
    }
}
