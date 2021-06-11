#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include "Evaluator.h"
#include "util.h"

using namespace std;

Evaluator :: Evaluator(){
	distanceMatrix = NULL;
	ratio = NULL;
}

Evaluator :: ~Evaluator(){
	if(distanceMatrix){
		for(int i = 0; i < numQueryObjects; i++)
			delete[] distanceMatrix[i];
		delete[] distanceMatrix;	
		distanceMatrix = NULL;
	}
	if(ratio)
		delete[] ratio;
}

void Evaluator :: parseClaTarget(char* filename){
	ifstream in(filename);
	
	string format;
	int version;
	
	in >> format >> version;
	skipline(in);
	
	int numClasses;
	in >> numClasses >> numObjects;
	skipline(in);
	
	//cout << format << " " << version << endl;
	//cout << numClasses << " " << numObjects << endl;
	classObject.resize(numObjects);
	
	int numTotalModels = 0;
	
	for(int i = 0; i < numClasses; i++){
		if(in.eof()){
			cout << "Error: the number of classes does not correspond with " << numClasses << endl;
			in.close();
			exit(EXIT_FAILURE);
		}
		string name;
		int parent, numM;
		in >> name >> parent >> numM;
		skipline(in);
		
		Clase c;
		c.setName(name);
		c.setParentClass(parent);
		
		for(int j = 0; j < numM; j++){
			int model;
			in >> model;
			if(in.fail()){
				cout << "Number of models does not correspond with " << numM << " in class " << name << endl;
				in.close();
				exit(EXIT_FAILURE);
			}
			
			skipline(in);
			
			c.addModel(model);
			classObject[model] = i;
			numTotalModels++;
		}
		classes.push_back(c);
	}
	
	if(numTotalModels!=numObjects){
		cout << "Num. of models does not correspond" << endl;
		in.close();
		exit(EXIT_FAILURE);
	}
		
	in.close();
}

void Evaluator :: parseClaQueries(char* filename){
	
	cout << filename << endl;
	
	ifstream in(filename);
	
	string format;
	int version;
	
	in >> format >> version;
	skipline(in);
	
	int numClasses, numQObjects;
	in >> numClasses >> numQObjects;
	skipline(in);
	
	//cout << format << " " << version << endl;
	//cout << numClasses << " " << numQObjects << endl;
	classQuery.resize(numQObjects);
	
	int numTotalModels = 0;
	
	for(int i = 0; i < numClasses; i++){
		if(in.eof()){
			cout << "Error: the number of classes does not correspond with " << numClasses << endl;
			in.close();
			exit(EXIT_FAILURE);
		}
		string name;
		int parent, numM;
		in >> name >> parent >> numM;
		skipline(in);
		
		//Clase c;
		//c.setName(name);
		//c.setParentClass(parent);
		
		for(int j = 0; j < numM; j++){
			int model;
			in >> model;
			//cout << name <<" "<<model << endl; 
			if(in.fail()){
				cout << "Number of models does not correspond with " << numM << " in class " << name << endl;
				in.close();
				exit(EXIT_FAILURE);
			}
			
			skipline(in);
			
			//c.addModel(model);
			classQuery[model] = i;
			numTotalModels++;
		}
		//classes.push_back(c);
	}
	
	if(numTotalModels!=numQObjects){
		cout << "Num. of models does not correspond" << endl;
		in.close();
		exit(EXIT_FAILURE);
	}
	in.close();
}

void Evaluator :: parseDistanceFile(char* filename){
	ifstream in(filename);
	int numModels;
	
	//in >> numQueryObjects >> numModels;
	numQueryObjects=7200;
	numModels=360;
	//skipline(in);
	
	if(numModels != numObjects){
		cout << "Number of objects from cla file and distance file does not correspond" << endl;
		in.close();
		exit(EXIT_FAILURE);
	}
	
	distanceMatrix = new double*[numQueryObjects];
	for(int i = 0; i < numQueryObjects; i++)
		distanceMatrix[i] = new double[numObjects];
	
	for(int i = 0; i < numQueryObjects; i++){
		for(int j = 0; j < numObjects; j++){
			in >> distanceMatrix[i][j];
		}
		//skipline(in);
	}
	
	in.close();
	
	//ofstream out("result.res");
	//for(int i = 0; i < 7200; i++){
	//	for(int j = 0; j < 360; j++){
	//		out << distanceMatrix[i][j] << " ";
	//	}
	//	out << endl;
	//}
	//out.close();
}

void Evaluator :: parsePartialityFile(const char* filename){
	int numElements;
	
	ifstream in(filename);
	in >> numElements;
	
	ratio = new double[numElements];
	for(int i = 0; i < numElements; i++){
		in >> ratio[i];
		skipline(in);
	}
	in.close();
}
void Evaluator :: printCla(){
	for(int i = 0; i < classes.size(); i++){
		cout << classes[i].getName() << " " << classes[i].getParentClass() << " ";
		vector<int> models = classes[i].getModels();
		cout << models.size() << endl;
		
		for(int j = 0; j < models.size(); j++){
			cout << models[j] << endl;
		}
	}	
}

void Evaluator :: getPrecisionRecallAll(EvaluationResult* er, int value){
	
	er->pr = new double[value + 1];
	memset(er->pr, 0, sizeof(double)*(value + 1));
	
	er->NN = 0.0;
	er->RP = 0.0;
	er->FT = 0.0;
	er->ST = 0.0;
	er->MAP = 0.0;
	double denom = 0.0;	
	for(int i = 0; i < numQueryObjects; i++){
		EvaluationResult aux;
		getPrecisionRecallPerModel(&aux, i, value);
		
		for(int j = 0; j <= value; j++)
			er->pr[j] = er->pr[j] + aux.pr[j];
		
		if(!partiality)
			er->NN += aux.NN;
		else
			er->NN += aux.NN * (1-ratio[i]);
		
		if(!partiality)
			er->RP += aux.RP;
		else
			er->RP += aux.RP * (1-ratio[i]);
		
		if(!partiality)
			er->FT += aux.FT;
		else
			er->FT += aux.FT * (1-ratio[i]);
		
		if(!partiality)
			er->ST += aux.ST;
		else
			er->ST += aux.ST * (1-ratio[i]);
		
		if(!partiality)	
			er->MAP += aux.MAP;
		else
			er->MAP += aux.MAP * (1-ratio[i]);
		
		if(partiality){
			denom = denom + (1 - ratio[i]);
		}else{
			denom = denom + 1;
		}
			
		delete[] aux.pr;

	}

	for(int i = 0; i <= value; i++){
		er->pr[i] /= denom;
	}
	
	er->NN /= denom;
	er->RP /= denom;
	er->FT /= denom;
	er->ST /= denom;
	er->MAP /= denom;
}

void Evaluator :: getPrecisionRecallPerModel(EvaluationResult* er, int model, int value){
	//Result * results;
	//results=(Result*)malloc(sizeof(Result));

	Result results[360];
	vector<double> precision_recall;
	double NN = 0.0;
	double RP = 1.0;
	double FT = 1.0; 
	double ST = 1.0;
	double MAP = 0.0;
	
	precision_recall.resize(value + 1);
	
	memset(results, 0, sizeof(Result)*numObjects);	
	
	//Extract the information for model
	for(int i = 0; i < numObjects; i++){
		results[i].id = i;
		results[i].classObject = classObject[i];
		results[i].distance = distanceMatrix[model][i];
	}
	
	//Sort by distance
	qsort(results, numObjects, sizeof(Result), compare);
	
	for(int i = 0; i < numObjects; i++){
		//cout << results[i].classObject << " "<< results[i].id << " " << results[i].distance << endl;
	}
	
	int count = 0;
	int j = 1; // We start in the second element, because the first element is the same model
	int queryClass = classQuery[model]; // class of the query
	int numModels = classes[queryClass].getNumberModels()-1; //Remaining models in the query's class
	
	//cout << "Clase:" << queryClass << endl;
	//cout << "Models:" << numModels << endl;
	
	//double precision[360][2];
	double** precision;
	precision=new double*[numModels];
	for (int i=0;i<numModels;i++)
		precision[i]=new double[2];
	
	while(count < numModels){
		if(results[j].classObject == queryClass){ //Compute precision-recall values
			if(j == 1)
				NN = 1.0;
			double rec = (double)(count + 1)/numModels;
			double prec = (double)(count + 1)/j;
			precision[count][0] = rec * 100; //To avoid round-off errors
			precision[count][1] = prec;
			MAP += prec;
			count++;
		}
		if(j == numModels){
			RP = (double)count / j;
			FT = RP;
		}
		if(j == 2*numModels){
			ST = (double)count / j;
		}
		
		j++;
	}
	
	MAP = MAP/numModels;
	int cont = numModels - 2;
	int recallValues = 100 - (100/value);
	double max = precision[cont + 1][1];
	int pos = value;
	precision_recall[pos] = max;
	
	pos--;
	while(cont>=0){
		if((int)(precision[cont][0]) >= recallValues){
			if(precision[cont][1] > max){
				max = precision[cont][1];
			}
			cont--;
		}else{
			precision_recall[pos] = max;
			recallValues = recallValues - value;
			pos--;
		}
	}
	
	while(pos>=0){
		precision_recall[pos] = max;
		pos--;
	}
	
	er->pr = new double[value + 1];
	
	for(int i = 0; i <= value; i++)
		er->pr[i] = precision_recall[i];
	
	er->NN = NN;
	er->RP = RP;
	er->FT = FT;
	er->ST = ST;
	er->MAP = MAP;

	//delete[]results;
	//free(results);
	for (int i=0;i<2;i++)
		delete[] precision[i];
}

void Evaluator :: getPrecisionRecallPerClass(EvaluationResult* er, int clase, int value){
	vector<int> models = classes[clase].getModels();
	
	er->pr = new double[value + 1];
	memset(er->pr, 0, sizeof(double)*(value+1));
	
	er->NN = 0.0;
	er->RP = 0.0;
	er->FT = 0.0;
	er->ST = 0.0;
	er->MAP = 0.0;
	double denom = 0.0;
	
	//precision_recall.resize(value + 1);
	
	for(int i = 0; i < models.size(); i++){
		EvaluationResult aux;
		getPrecisionRecallPerModel(&aux, models[i], value);
		
		for(int j = 0; j <= value; j++)
			er->pr[j] = er->pr[j] + aux.pr[j];
		
		if(!partiality)
			er->NN += aux.NN;
		else
			er->NN += aux.NN * (1-ratio[i]);
		
		if(!partiality)
			er->RP += aux.RP;
		else
			er->RP += aux.RP * (1-ratio[i]);
		
		if(!partiality)
			er->FT += aux.FT;
		else
			er->FT += aux.FT * (1-ratio[i]);
		
		if(!partiality)
			er->ST += aux.ST;
		else
			er->ST += aux.ST * (1-ratio[i]);
		
		if(!partiality)	
			er->MAP += aux.MAP;
		else
			er->MAP += aux.MAP * (1-ratio[i]);
		
		if(partiality){
			denom = denom + (1 - ratio[i]);
		}else{
			denom = denom + 1;
		}
		delete[] aux.pr;
	}
	
	for(int i = 0; i <= value; i++){
		er->pr[i] /= denom;
	}
	
	er->NN /= denom;
	er->RP /= denom;
	er->FT /= denom;
	er->ST /= denom; 
	er->MAP /= denom;
}
