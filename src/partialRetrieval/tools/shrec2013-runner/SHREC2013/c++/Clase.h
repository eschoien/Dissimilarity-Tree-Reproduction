#ifndef __CLASE_H__
#define __CLASE_H__

#include <vector>
#include <string>
using namespace std;

class Clase{
		string name;
		int parentClass;
		vector<int> models;
		
	public:
		Clase(){}
		void setName(string n){name = n;}
		void setParentClass(int num){ parentClass = num;}
		void addModel(int num){models.push_back(num);}
		
		string getName(){return name;}
		int getParentClass(){ return parentClass;}
		vector<int> getModels(){return models;}
		int getNumberModels(){ return models.size();}
};

#endif
