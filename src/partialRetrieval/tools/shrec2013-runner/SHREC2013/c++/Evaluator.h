#ifndef __EVALUATOR_H__
#define __EVALUATOR_H__

#include <vector>
#include "Clase.h"
using namespace std;

struct Result{
    int id;				//Identificador del objeto
    int classObject;	//Clase del objeto
	double distance;	//Distancia hacia el objeto consulta
};

struct EvaluationResult{
	double* pr;
	double RP;
	double FT;
	double ST;
	double EM;
	double DGC;
	double NN;
	double MAP;
};

class Evaluator{
	vector<Clase> classes;
	vector<int> classObject;
	vector<int> classQuery;
	
	double** distanceMatrix;
	int numObjects;
	int numQueryObjects;
	int partiality;
	double* ratio;
	
	public:
		Evaluator();
		~Evaluator();
		
		void parseClaTarget(char* filename);
		void parseClaQueries(char* filename);
		void parseDistanceFile(char* filename);
		void parsePartialityFile(const char* filename);
		void printCla();
		
		void setPartiality(int part){partiality = part;}
		/*Methods for calculating performance measures*/
		void getPrecisionRecallAll(EvaluationResult* er, int value = 10);
		void getPrecisionRecallPerModel(EvaluationResult *er, int model, int value = 10);
		void getPrecisionRecallPerClass(EvaluationResult *er, int clase, int value = 10);
};
#endif
