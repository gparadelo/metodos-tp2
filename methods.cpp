#include "methods.h"



int imageMatrix::kNearestNeighbors(const vector<uchar*> &newFace, int k) {
    vector< pair <int, int> > distances;

    for (int i = 0; i < faceMatrix.size(); ++i) {
        pair<int, int> dist (getSquaredNorm(faceMatrix[i], newFace), people[i]);
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


int imageMatrix::getSquaredNorm(const vector<uchar*> &v1,const vector<uchar*> &v2) {
    int distance = 0;

    for (int i = 0; i < v1.size(); ++i) {
        distance += ((int)v1[i] - (int)v2[i]) ^ 2;  //Cuidado que la resta de uchar puede dar cualquier cosa.
    }

    return distance;
}


bool imageMatrix::pairCompare(pair<uchar*, int> i, pair<uchar*, int> j) { return (i.first < j.first); }