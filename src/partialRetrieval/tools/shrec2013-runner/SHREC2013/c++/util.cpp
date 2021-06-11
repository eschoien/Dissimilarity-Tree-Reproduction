#include <iostream>
#include <cmath>
#include "util.h"
#include "Evaluator.h"

using namespace std;

void skipline(istream &in){
	char c;
	while(in>>noskipws>>c && c!='\n');
	in>>skipws;
}

int compare(const void *a, const void *b){
    Result *a1 = (Result*)a;
    Result *b1 = (Result*)b;

    if(a1->distance > b1->distance)
        return 1;
    else if(a1->distance < b1->distance)
        return -1;
    else
        return 0;
}
