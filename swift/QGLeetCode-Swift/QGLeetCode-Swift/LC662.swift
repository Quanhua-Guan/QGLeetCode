//
//  LC662.swift
//  QGLeetCode-Swift
//
//  Created by 宇园 on 2022/8/27.
//

import Foundation

class LC662 {
    
    public class func test() {
        let root = LC662.TreeNode(0, LC662.TreeNode(0), LC662.TreeNode(0))
        var left = root.left!
        var right = root.right!
        for _ in 0..<25 {
            left.right = LC662.TreeNode(0)
            left = left.right!
            
            right.left = LC662.TreeNode(0)
            right = right.left!
        }
        
        print(LC662.Solution().widthOfBinaryTree(root))
    }
    
    /**
     * Definition for a binary tree node.
     */
    public class TreeNode {
        public var val: Int
        public var left: TreeNode?
        public var right: TreeNode?
        public init() { self.val = 0; self.left = nil; self.right = nil; }
        public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
        public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
            self.val = val
            self.left = left
            self.right = right
        }
    }
    
    class Solution {
                
        func widthOfBinaryTree(_ root: TreeNode?) -> Int {
            var queue: [(node: TreeNode, index: Int)] = []
            queue.append((root!, 1))
            
            var maxCount = 0
            
            while !queue.isEmpty {
                
                var nextQueue: [(node: TreeNode, index: Int)] = []
                for info in queue {
                    if info.node.left != nil {
                        nextQueue.append((info.node.left!, info.index &* 2))
                    }
                    if info.node.right != nil {
                        nextQueue.append((info.node.right!, info.index &* 2 &+ 1))
                    }
                }
                maxCount = max(maxCount, queue.last!.index &- queue.first!.index &+ 1)
                
                queue = nextQueue
            }
            
            return maxCount
        }
        
    }
}
