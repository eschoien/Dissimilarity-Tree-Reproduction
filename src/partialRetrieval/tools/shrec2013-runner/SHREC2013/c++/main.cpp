#include <iostream>
#include <cstdlib>
#include <cstring>
#include "Evaluator.h"

using namespace std;

int main(int argc, char* argv[]){
	if(argc < 7){
		cout << "Too few input parameters" << endl;
		cout << "evaluator2 target.cla query.cla file.res -bin value [-model | -class | -all] value [partiality]" << endl;
		exit(EXIT_FAILURE);
	}
	
	char* fileclatrain = argv[1];
	char* fileclatest = argv[2];
	char* fileres = argv[3];
	string bin = argv[4];
	if(bin.compare("-bin")!=0){
		cout << "Option bin must be set" << endl;
		exit(EXIT_FAILURE);
	}
	
	int numBin = atoi(argv[5]);
	
	string option = argv[6];
	int opt = 0;
	
	if(option.compare("-model")==0)
		opt = 1;
	else if(option.compare("-class")==0)
		opt = 2;
	else if(option.compare("-all")==0)
		opt = 3;
	else{
		cout << "Option is unknown. It should be (-model, -class, or -all)" << endl;
		exit(EXIT_FAILURE);
	}
	
	if(opt == 1 || opt == 2){
		if(argc < 8){
			cout << "Number of parameters is wrong." << endl;
			exit(EXIT_FAILURE);
		}
	}
	
	int param;
	int partiality = 0;
	
	if(argc == 8){
		param = atoi(argv[7]);
	}
	
	if(argc == 9){
		param = atoi(argv[7]);
		partiality = atoi(argv[8]);
	}
	
	cout << partiality << endl;
	Evaluator e;
	e.setPartiality(partiality);
	//cout << fileclatrain << endl;
	//cout << fileclatest << endl;
	
	e.parseClaTarget(fileclatrain); cout << "Target loaded" << endl;
	e.parseClaQueries(fileclatest); cout << "Queries loaded" << endl;
	e.parseDistanceFile(fileres);	cout << "Distances loaded" << endl;
	
	if(partiality){
		e.parsePartialityFile("partiality.txt"); cout << "Partiality loaded" << endl;
	}
	
	EvaluationResult er;
	memset(&er, 0, sizeof(EvaluationResult));
	switch(opt){
		case 1: e.getPrecisionRecallPerModel(&er, param, numBin);
				break;
		case 2: e.getPrecisionRecallPerClass(&er, param, numBin);
				break;
		case 3: e.getPrecisionRecallAll(&er, numBin);
				break;
	}
	
	cout << "Precision-Recall" << endl;
	for(int i = 0; i <= numBin; i++){
		cout << er.pr[i] << endl;
	}
	
	cout << "NN = " << er.NN << endl;
	cout << "RP = " << er.RP << endl;
	cout << "FT = " << er.FT << endl;
	cout << "ST = " << er.ST << endl;
	cout << "MAP= " << er.MAP << endl;
	delete[] er.pr;
	return EXIT_SUCCESS;
}
