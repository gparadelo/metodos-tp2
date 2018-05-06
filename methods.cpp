#include "methods.h"

vector<vector<int>> createTrainMatrix(string trainFaces) {


}


vector<int> getPeople(string trainFaces) {


}


int kNearestNeighbors(vector <vector <int> > trainMatrix, vector<int> people, vector<int> newFace, int k) {
    vector< pair <int, int> > distances;

    for (int i = 0; i < trainMatrix.size(); ++i) {
        pair<int, int> dist (getSquaredNorm(trainMatrix[i], newFace), people[i]);
        distances.push_back(dist);
    }

    vector<int> neighbors;
    sort(distances.begin(), distances.end(), pairCompare);

    for (int i = 0; i < k; ++i) {
        neighbors.push_back(distances[i].second);
    }

    int nearest = neighbors[0];
    int maxCount = count(neighbors.begin(), neighbors.end(), neighbors[0]);
    for (int i = 1; i < neighbors.size(); ++i) {
        int newCount = count(neighbors.begin(), neighbors.end(), neighbors[i]);
        if (newCount > maxCount) {
            maxCount = newCount;
            nearest = neighbors[i];
        }
    }

    return nearest;
}


int getSquaredNorm(vector<int> v1,vector<int> v2) {
    int distance = 0;

    for (int i = 0; i < v1.size(); ++i) {
        distance += ((v1[i] - v2[i]) ^ 2);
    }

    return distance;
}


bool pairCompare(pair<int, int> i, pair<int, int> j) { return (i.first < j.first); }