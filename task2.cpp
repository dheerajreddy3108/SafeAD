#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

/* In machine learning, we sometimes have to find the nearest neighbour for a
 * given point. This can be efficiently done using k-d trees. In this task,
 * you will implement a nearest-neighbour search in euclidean space using a
 * k-d tree.
 *
 * Compile with g++ -std=c++17 task2.cpp -o task2 */

constexpr int pointDims = 3;
typedef std::array<float, pointDims> Point;

struct Node {
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;
    Point point;

    Node(Point &pt) : left(nullptr), right(nullptr), point(pt) { }
};

class KDTree {
public:
    KDTree(std::vector<Point> &points) {
        root_ = constructTree(points, 0);
    }

    Point nearestNeighbour(Point queryPoint) const {
        auto [node, dist] = nearestPointInSubtree(queryPoint, root_, 0);
        return node->point;
    }

private:
    std::shared_ptr<Node> constructTree(std::vector<Point> &points, int depth) {
        if (points.size() == 0) {
            return nullptr;
        }

        int axis = depth % pointDims;
        std::sort(points.begin(), points.end(),
                  [&] (const Point &a, const Point &b) {
                      return a[axis] < b[axis];
                });
        int medianIdx = points.size() / 2;
        auto node = std::make_shared<Node>(points[medianIdx]);
        std::vector<Point> leftPts(medianIdx);
        std::copy(points.begin(), points.begin() + medianIdx, leftPts.begin());
        std::vector<Point> rightPts(points.size() - medianIdx - 1);
        std::copy(points.begin() + medianIdx + 1, points.end(), rightPts.begin());
        node->left = constructTree(leftPts, depth + 1);
        node->right = constructTree(rightPts, depth + 1);
        return node;
    }

    std::tuple<std::shared_ptr<Node>, float> nearestPointInSubtree(
        Point queryPoint, std::shared_ptr<Node> rootNode, int depth) const {

        int axis = depth % pointDims;
        float axisDistToRoot = queryPoint[axis] - rootNode->point[axis];
        float distToRoot = 0.0;
        for (int i = 0; i < pointDims; ++i) {
            distToRoot += std::pow(queryPoint[i] - rootNode->point[i], 2.0);
        }
        distToRoot = std::pow(distToRoot, 0.5);

        if (!rootNode->left && !rootNode->right) {
            return {rootNode, distToRoot};
        }

        

        /**********************************************************************
         * TODO: Implement the remaining part of this function by calling it
         * recursively. Also take care of the edge cases (e.g. one child node).
         * This can be done as follows:
         * 1: Figure out on which side of the separating hyperplane the query
         *    point lies and recursively descend into that part of the tree.
         * 
         * interpreted as similar to tree building
         * 
         * 2: If the distance of the query point to the best point in that
         *    sub-tree is smaller than the distance to the separating hyperplane
         *    (axisDistToRoot), just return the best point and corresponding
         *    distance from that sub-tree.
         * 
         * calcualte distance between the points and query point
         * 
         * 3: Otherwise the closest point to the query point can also be the
         *    root node or a point in the other sub-tree. Calculate all
         *    distances and return the node with the smallest distance to the
         *    query point.
         **********************************************************************/

        // code added starts from here
      
        //float best_dist = 0.0;
        
        for(int axis = 0;axis<pointDims; axis++){
            if(queryPoint[axis]<rootNode->point[axis]){
            // must search on left side
            axisDistToRoot = queryPoint[axis] - rootNode->left->point[axis];
            float distToRoot = 0.0;
            for (int i = 0; i < pointDims; ++i) {
                distToRoot += std::pow(queryPoint[i] - rootNode->left->point[i], 2.0);
            }
            distToRoot = std::pow(distToRoot, 0.5);
            std::shared_ptr<Node> best_node = rootNode->left;
            if (distToRoot<axisDistToRoot){
                return {best_node, axisDistToRoot};
                }
            else{
                float distToRoot = 0.0;
                for (int i =0; i<pointDims;i++){
                    distToRoot += std::pow(queryPoint[i]-rootNode->point[i], 2.0);

                }
                distToRoot = std::pow(distToRoot, 0.5);

            }
            }
            else{
            //search on right side
            axisDistToRoot = queryPoint[axis] - rootNode->right->point[axis];
            float distToRoot = 0.0;
            for (int i = 0; i < pointDims; ++i) {
                distToRoot += std::pow(queryPoint[i] - rootNode->right->point[i], 2.0);
            }
            distToRoot = std::pow(distToRoot, 0.5);
            std::shared_ptr<Node> best_node = rootNode->right;
            ;
            if (distToRoot<axisDistToRoot){
                return {best_node, axisDistToRoot};
                }
            else{
                float distToRoot = 0.0;
                for (int i =0; i<pointDims;i++){
                    distToRoot += std::pow(queryPoint[i]-rootNode->point[i], 2.0);

                }
                distToRoot = std::pow(distToRoot, 0.5);

            }
            }

        }

        

       


    }

    std::shared_ptr<Node> root_;

};

void printPoint(Point &point) {
    std::cout << "(";
    for (int i = 0; i < pointDims; ++i) {
        std::cout << point[i];
        if (i + 1 < pointDims) {
            std::cout << ", ";
        }
    }
    std::cout << ")" << std::endl << std::endl;
};

int main() {
    // Insert some points to the tree
    std::vector<Point> points {
        {0.935, 0.086, 0.069}, 
        {0.407, 0.5  , 0.349}, 
        {0.959, 0.394, 0.004}, 
        {0.418, 0.608, 0.452},
        {0.331, 0.704, 0.418}, 
        {0.76 , 0.988, 0.544},
        {0.89 , 0.063, 0.137}, 
        {0.574, 0.903, 0.101},
        {0.9  , 0.889, 0.708}, 
        {0.322, 0.963, 0.816} 
    };

    KDTree tree(points);

    // Find nearest neighbour to a given point
    Point nn = tree.nearestNeighbour({0.5, 0.5, 0.5});
    std::cout << "I found the following nearest neighbour for (0.5, 0.5, 0.5),"
              << std::endl << "[expected (0.418, 0.608, 0.452)]:" << std::endl;
    printPoint(nn);

    nn = tree.nearestNeighbour({0.2, 0.7, 0.8});
    std::cout << "I found the following nearest neighbour for (0.2, 0.7, 0.8),"
              << std::endl << " [expected (0.322, 0.963, 0.816)]:" << std::endl;
    printPoint(nn);

    nn = tree.nearestNeighbour({0.7, 0.2, 0.5});
    std::cout << "I found the following nearest neighbour for (0.7, 0.2, 0.5),"
              << std::endl << " [expected (0.89 , 0.063, 0.137)]:" << std::endl;
    printPoint(nn);
    return 0;
}