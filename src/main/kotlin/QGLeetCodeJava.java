package com.cqmh.qgleetcode;

import java.util.HashMap;

public class QGLeetCodeJava {

}

// Definition for a binary tree node.
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
    }
}

/// 面试题 04.06. 后继者
class SolutionMs0406 {
    public TreeNode mostLeft(TreeNode root) {
        TreeNode left = root;
        while (left != null && left.left != null) {
            left = left.left;
        }
        return left;
    }

    public TreeNode pFatherNode(TreeNode root, TreeNode p, HashMap<TreeNode, TreeNode> mem) {
        if (mem.containsKey(p)) {
            return mem.get(p);
        }

        TreeNode pFather = null;
        TreeNode current = root;
        if (root != p) {
            while (current != null) {
                if (current.val > p.val) {
                    // 往左搜索
                    if (current.left != null) {
                        if (current.left == p) {
                            pFather = current;
                            break;
                        } else {
                            mem.put(current.left, current);
                            current = current.left;
                        }
                    } else {
                        break;
                    }
                } else { // current.val < p.val
                    // 往右搜索
                    if (current.right != null) {
                        if (current.right == p) {
                            pFather = current;
                            break;
                        } else {
                            mem.put(current.right, current);
                            current = current.right;
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        mem.put(p, pFather);
        return pFather;
    }

    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        if (p == null) return null;

        HashMap<TreeNode, TreeNode> mem = new HashMap<>();

        // 目标：找到第一个大于 p 的节点

        //（1）p 有右子树，则必然存在『下一个』节点
        if (p.right != null) {
            // 如果 p 的右子树不为空，则 p 的后继即为其右子树的最小值
            return mostLeft(p.right);
        }

        //（2）p 没有右子树

        //（2.1）p 没有右子树，没有父节点，则它没有『下一个』
        if (p == root) {
            return null;
        }

        //（2.1）p 没有右子树，有父节点，则它【可能】有『下一个』
        // 先找到 p 的父节点（顺便缓存，加速）
        TreeNode pFather = pFatherNode(root, p, mem); // pFather != null

        //（2.1.1）假如 p 是其父节点的左子节点，则父节点即为『下一个』
        if (pFather.left == p) {
            return pFather;
        }

        //（2.1.2）假如 p 是其父节点的右子节点，

        // 则找到父节点的父节点
        // 如果【父节点的父节点】的【左节点】是【父节点】，则【父节点的父节点】即为『下一个』，

        // 否则, 继续一直往上找（直到root），
        // - 如果找到一个【p的祖先节点1】是另一个【p的祖先节点2】的【左子节点】，则【p的祖先节点2】即为『下一个』
        // - 如果最终没找到符合条件的节点，则没有『下一个』
        TreeNode node = null;
        do {
            node = pFather;
            pFather = pFatherNode(root, pFather, mem);
        } while (pFather != null && pFather.left != node);

        return pFather;
    }

    public TreeNode inorderSuccessor_Better(TreeNode root, TreeNode p) {
        if (p == null) return null;

        // 目标：找到第一个大于 p 的节点

        //（1）p 有右子树，则必然存在『下一个』节点
        if (p.right != null) {
            // 如果 p 的右子树不为空，则 p 的后继即为其右子树的最小值
            return mostLeft(p.right);
        }

        //（2）p 没有右子树
        TreeNode theAncestor = null;
        TreeNode current = root;
        while (current != null) {
            // 以 current 为分界
            if (current.val > p.val) {
                theAncestor = current; // current 可能是『下一个』
                current = current.left;
            } else { // current.val > p.val
                // current 不可能是『下一个』，current.right 可能是『下一个』
                current = current.right;
            }
        }

        return theAncestor;
    }
}
